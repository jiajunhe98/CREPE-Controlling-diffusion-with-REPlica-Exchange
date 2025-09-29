import argparse
import pickle, torch, PIL.Image
from pathlib import Path
import dnnlib
import numpy as np
from tqdm import tqdm
import json
import os
import clip
import ImageReward as RM
import datetime

torch.manual_seed(20)
device = torch.device('cuda:0')

reward_model = RM.load("ImageReward-v1.0").to(device)
reward_model.eval()
print(f"Reward model device: {next(reward_model.parameters()).device}")

clip_model_name = "ViT-L/14@336px"  # match CLIPScore paper default
clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
clip_model.eval()



def cuda_mem(tag=""):
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available"); return
    torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    MB = 1024**2
    alloc   = torch.cuda.memory_allocated(dev)    / MB
    reserv  = torch.cuda.memory_reserved(dev)     / MB
    peak    = torch.cuda.max_memory_allocated(dev)/ MB
    print(f"[{tag}] alloc={alloc:.1f} MiB  reserved={reserv:.1f} MiB  peak={peak:.1f} MiB")


def tile_images(x, w, h):
    assert x.ndim == 4 # NCHW => CHW
    return x.reshape(h, w, *x.shape[1:]).permute(2, 0, 3, 1, 4).reshape(x.shape[1], h * x.shape[2], w * x.shape[3])


@torch.inference_mode()
def compute_clip_and_imagereward(
    images: torch.Tensor,
    prompts: list[str],
    device,
    batch: int = 64,
    want_clip: bool = True,
    want_imr: bool = True,
):
    """
    images: [B, PT, C, H, W] uint8 (CUDA)
    returns (clip_scores[B,PT] or None, imr_scores[B,PT] or None) on CPU
    """
    assert images.is_cuda and images.dtype == torch.uint8
    B, PT, C, H, W = images.shape
    assert len(prompts) == B
    N = B * PT

    # Text features once
    if want_clip:
        tokens = clip.tokenize(prompts, context_length=77, truncate=True).to(device)
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        txt_flat = txt.repeat_interleave(PT, dim=0)  # [N, D]

        # robust dtype for CLIP visual (works for ViT / RN)
        clip_dtype = next(clip_model.visual.parameters()).dtype
        clip_buf = torch.empty(N, dtype=torch.float32)  # CPU buffer

    imr_buf = torch.empty(N, dtype=torch.float32) if want_imr else None

    flat = images.reshape(N, C, H, W)

    for i in range(0, N, batch):
        j = min(i + batch, N)

        # one D2H copy + PIL construction
        np_hwc = flat[i:j].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        pils = [PIL.Image.fromarray(arr) for arr in np_hwc]

        if want_clip:
            x = torch.stack([clip_preprocess(p) for p in pils]).to(device)
            x = x.to(dtype=clip_dtype)
            img = clip_model.encode_image(x)
            img = img / img.norm(dim=-1, keepdim=True)
            clip_buf[i:j] = (100.0 * (img * txt_flat[i:j]).sum(dim=-1)).detach().cpu()

        if want_imr:
            # ImageReward exposes .score(prompt, PIL.Image)
            for k, pil in enumerate(pils):
                b, _t = divmod(i + k, PT)
                imr_buf[i + k] = float(reward_model.score(prompts[b], pil))

    clip_scores = clip_buf.view(B, PT) if want_clip else None
    imr_scores  = imr_buf.view(B, PT)  if want_imr  else None
    return clip_scores, imr_scores


@torch.no_grad()
def build_prompts_from_imagenet_labels(labels_onehot: torch.Tensor, idx_to_label):
    """
    labels_onehot: [B,D] one-hot or None
    returns list[str] of length B using 'a photo of a {classname}'.
    """
    B, D = labels_onehot.shape
    cls = labels_onehot.argmax(dim=-1).reshape(-1).tolist()  # [N]
    return [f"{idx_to_label[i]}" for i in cls]


@torch.no_grad()
def get_model(img_net, verbose=False):

    if img_net == 512:
        model_url = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.135.pkl'
    elif img_net == 64:
        model_url = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl'

    with dnnlib.util.open_url(model_url, verbose=verbose) as f:
        data = pickle.load(f)
    net = data['ema'].to(device).eval()
    encoder = data['encoder']
    assert encoder is not None
    encoder.init(device)
    return net, encoder


@torch.no_grad()
def get_imagenet_labels():
    """Downloads and returns the ImageNet class index to name mapping."""
    url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    print(f'Loading ImageNet class map from {url}...')
    with dnnlib.util.open_url(url, verbose=True) as f:
        labels = json.load(f)
    # The labels are stored with string keys, so we create a simple list
    idx_to_label = [labels[str(k)][1] for k in range(1000)]
    return idx_to_label


class BatchedEDM(torch.nn.Module):
    """
    A wrapper for the EDM model that automatically handles multiple batch dimensions.
    
    This allows you to pass inputs with shapes like [B1, B2, C, H, W]
    without needing to manually flatten and reshape them.
    """
    def __init__(self, model):
        super().__init__()
        # Store the original EDM model
        self.model = model

    # --- FIX: Expose the necessary attributes from the original model ---
    @property
    def img_resolution(self):
        return self.model.img_resolution

    @property
    def img_channels(self):
        return self.model.img_channels

    @property
    def label_dim(self):
        return self.model.label_dim

    def forward(self, x, t, labels=None, **kwargs):
        # 1. Store the original shape and number of batch dimensions
        original_shape = x.shape
        num_batch_dims = x.ndim - 3 # The last 3 dims are (C, H, W)
        
        # If there's only one batch dimension, we don't need to do anything
        if num_batch_dims <= 1:
            return self.model(x, t, labels, **kwargs)
            
        # 2. Flatten the batch dimensions
        # Reshape x from [B1, B2, ..., C, H, W] to [B, C, H, W]
        flattened_x = x.reshape(-1, *original_shape[num_batch_dims:])
        
        # 3. Flatten the time and label tensors to match
        # Reshape t from [B1, B2, ...] to [B]
        flattened_t = t.reshape(-1)
        
        flattened_labels = None
        if labels is not None:
            # Reshape labels from [B1, D] to [B, D]
            # print(labels.shape)
            flattened_labels = labels.unsqueeze(1).expand(-1, x.shape[1], -1).reshape(-1, labels.shape[-1])
            
                
        # 4. Call the original model with the flattened tensors
        # print(flattened_x.shape, flattened_t.shape, flattened_labels.shape if flattened_labels is not None else None)
        output = self.model(flattened_x, flattened_t, flattened_labels, **kwargs)
        
        # 5. Reshape the output back to the original multi-batch dimension shape
        # The output has the same C, H, W as the input, so we can use their shapes.
        output_shape = (*original_shape[:num_batch_dims], *output.shape[1:])
        reshaped_output = output.reshape(output_shape)
        
        return reshaped_output


# --- Wrapper for the Encoder/Decoder ---
class BatchedEncoder:
    """A wrapper for an Encoder that automatically handles multiple batch dimensions."""
    def __init__(self, encoder):
        self.encoder = encoder
    
    def _apply_with_batching(self, func, x):
        original_shape = x.shape
        # Latents are (B, C, H, W), pixels are (B, C, H, W)
        # The number of data dimensions is always 3.
        num_batch_dims = x.ndim - 3

        if num_batch_dims <= 1:
            return func(x)

        # Flatten all batch dimensions into one
        flattened_x = x.reshape(-1, *original_shape[num_batch_dims:])
        # Apply the original function
        output = func(flattened_x)

        # Reshape the output back to the original batch structure
        output_shape = (*original_shape[:num_batch_dims], *output.shape[1:])
        reshaped_output = output.reshape(output_shape)
        return reshaped_output
        
    def decode(self, latents):
        return self._apply_with_batching(self.encoder.decode, latents)
        
    def encode_latents(self, latents):
        return self._apply_with_batching(self.encoder.encode_latents, latents)

    def encode_pixels(self, pixels):
        return self._apply_with_batching(self.encoder.encode_pixels, pixels)
        
    def encode(self, pixels):
        return self._apply_with_batching(self.encoder.encode, pixels)


@torch.no_grad()
def log_norm_prob(x, mu, std):
    # expand std to match x shape
    std = std.expand(x.shape)
    return -0.5 * ((x - mu) / std).pow(2).sum(dim=(-3, -2,-1)) - (std.log() + np.log(2*np.pi)/2).sum(dim=(-3, -2,-1))


@torch.no_grad()
def edm_sampler(net, inital, t_steps, labels=None, guidance=1.7):
    def denoise(x, t):
        Dx = net(x, t, labels).clone()
        if guidance == 1.0 or labels is None:
            return Dx
        ref_Dx = net(x, t, None)
        return ref_Dx.lerp(Dx, guidance)
    
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    x = inital
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        score = (-x + denoise(x, t_cur)) / t_cur**2  
        x = x + t_cur * (t_cur -  t_next / 2) * score + torch.sqrt(t_next**2 - t_next**2 / 4) * torch.randn_like(x)
    return x


@torch.no_grad()
def edm_sampler_for_initialization(net, noise, labels, t_steps, guidance=1.7):
    initial = noise * t_steps[0]
    x = edm_sampler(net, initial, t_steps, labels=labels, guidance=guidance)
    #x = x + t_steps.reshape(1, -1, 1, 1, 1) * torch.randn_like(x)
    return x


# --- Accelerated Parallel Tempering Sampler ---
@torch.no_grad()
def apt_sampler(
    net, 
    B, 
    T, 
    labels, 
    shape, 
    num_pt_iterations, 
    reward_min,
    reward_max,
    reward_rho, 
    encoder,
    prompts,
    sigma_min=0.002, 
    sigma_max=80.0, 
    rho=7.0, 
    guidance=1.7, 
    guidance_target=1.7,
    init_guidance=1.7, 
    switch_t_idx=0, 
    local_exploration=False, 
    mask_last=0
):
    t_schedule = (sigma_max**(1/rho) + torch.linspace(0, 1, T, device=device) * (sigma_min**(1/rho) - sigma_max**(1/rho))).pow(rho)
    
    initial_noise = torch.randn(B, T, *shape, device=device)
    print("Initializing chains.")
    # We init using a sample per noise level
    xs = edm_sampler_for_initialization(net, initial_noise, labels, t_schedule, guidance=init_guidance)
    xs = xs + t_schedule.reshape(1, -1, 1, 1, 1) * torch.randn_like(xs)
    print("Chains initialized.")
    print(f"Switching from APT to EDM at time {t_schedule[T-1-switch_t_idx]}.")
    xs = xs[:,:T-switch_t_idx]

    results = torch.empty(B, num_pt_iterations, *shape, device=device)
    accept_probs = torch.zeros(B, T-1-switch_t_idx, device=device)

    print("Starting APT")
    for pt_iteration in tqdm(range(num_pt_iterations)):
        start_idx  = pt_iteration%2
        xs, mask = apt_iteration(
            xs, 
            t_schedule[:T-switch_t_idx], 
            labels, 
            net, 
            start_idx, 
            sigma_max, 
            guidance, 
            guidance_target, 
            reward_min,
            reward_max,
            reward_rho, 
            encoder,
            prompts,
            local_exploration=local_exploration, 
            mask_last=0)
        results[:, pt_iteration] = xs[:, -1].clone()
        accept_probs[:, start_idx::2] += mask

    accept_probs /= num_pt_iterations / 2
    final = edm_sampler(net, results, t_schedule[T-switch_t_idx-1:], labels=labels, guidance=guidance)
    return final, accept_probs


@torch.no_grad()
def apt_iteration(
    xs, 
    temp_schedule, 
    labels, 
    net, 
    start_idx, 
    sigma_max, 
    guidance, 
    guidance_target,
    reward_min,
    reward_max,
    reward_rho, 
    encoder,
    prompts,
    local_exploration=False, 
    mask_last=False
):
    """
    Modifies xs in-place.
    """
    B, T = xs.shape[0], xs.shape[1]

    # --- Define reward annealing ---
    reward_schedule = reward_min ** (1/reward_rho) + torch.linspace(0, 1, T, device=device) * (reward_max ** (1/reward_rho) - reward_min ** (1/reward_rho))
    reward_schedule = reward_schedule ** reward_rho * 100

    # --- Batched Local Exploration ---
    if local_exploration:
        denoised_cond = net(xs,  temp_schedule.unsqueeze(0).expand(B, -1), labels).clone()
        denoised_uncond = net(xs,  temp_schedule.unsqueeze(0).expand(B, -1), None).clone()
        denoised_sampling = guidance_target * denoised_cond + (1-guidance_target) * denoised_uncond # Check that not guidance
        score_sampling = (-xs + denoised_sampling) / temp_schedule.reshape(1, -1, 1, 1, 1)**2 
        lang_std = torch.sqrt(2 * temp_schedule[:-1] * (temp_schedule[:-1] - temp_schedule[1:])).reshape(1, -1, 1, 1, 1)
        xs[:, :-1] = xs[:, :-1] + 0.5 * lang_std**2 * score_sampling[:, :-1] + lang_std * torch.randn_like(xs[:, :-1]) # No langevin at lowest time

    # --- Resample Highest Time ---
    xs[:, 0] = sigma_max * torch.randn_like(xs[:, 0])

    # --- APT ---
    x1s = xs[:, start_idx:-1:2]
    x0s = xs[:, start_idx+1::2]
    idx1s = torch.arange(T)[start_idx:-1:2]
    idx0s = torch.arange(T)[start_idx+1::2]
    t1s = temp_schedule[idx1s]  # t1s > t0s
    t0s = temp_schedule[idx0s]
    dt = (t1s - t0s).abs()

    r1s = reward_schedule[idx1s]
    r0s = reward_schedule[idx0s]

    t1s_b = t1s.reshape(1, -1, 1, 1, 1)
    t0s_b = t0s.reshape(1, -1, 1, 1, 1)
    dt_b = dt.reshape(1, -1, 1, 1, 1)

    # Denoising std
    bwd_std = torch.sqrt(2 * t1s_b * dt_b)

    # Noising std
    fwd_std = torch.sqrt(2 * t0s_b * dt_b)        

    # x1s, idx1s, t1s represent larger time (more noise)
    # x0s, idx0s, t0s represent smaller time (closer to data)

    # --- Generate candidate samples x0s_cand from x1s ---
    denoised_x1s_uncond = net(x1s,  t1s.unsqueeze(0).expand(B, -1), None).clone()
    denoised_x1s_cond = net(x1s,  t1s.unsqueeze(0).expand(B, -1), labels).clone()
    denoised_x1s = guidance * denoised_x1s_cond + (1-guidance) * denoised_x1s_uncond
    denoised_x1s_image = encoder.decode(denoised_x1s)
    reward_x1s = compute_clip_and_imagereward(denoised_x1s_image, prompts, device=device, want_clip=False, want_imr=True)[1].to(device)
    reward_x1s = reward_x1s * r1s

    denoised_cond = net(x1s,  t1s.unsqueeze(0).expand(B, -1), labels).clone()
    score_cond = (-x1s + denoised_cond) / t1s_b**2 
    denoised_uncond = net(x1s,  t1s.unsqueeze(0).expand(B, -1), None).clone()
    score_uncond = (-x1s + denoised_uncond) / t1s_b**2 
    denoised_sampling = guidance * denoised_cond + (1-guidance) * denoised_uncond
    score_sampling = (-x1s + denoised_sampling) / t1s_b**2 
    
    x0s_cand = x1s + bwd_std**2 * score_sampling + bwd_std * torch.randn_like(x1s)

    # Diffusion forward process
    dm_fwd = log_norm_prob(x1s, x0s_cand, fwd_std) 

    # Uncond diffusion backward process
    bwd_mean_uncond = x1s + bwd_std**2 * score_uncond
    dm_bwd_uncond = log_norm_prob(x0s_cand, bwd_mean_uncond, bwd_std)

    # Cond diffusion backward process
    bwd_mean_cond = x1s + bwd_std**2 * score_cond
    dm_bwd_cond = log_norm_prob(x0s_cand, bwd_mean_cond, bwd_std)

    # Sampling forward process
    sample_fwd = log_norm_prob(x1s, x0s_cand, fwd_std) 

    # Sampling backward process
    bwd_mean = x1s + bwd_std**2 * score_sampling 
    sample_bwd = log_norm_prob(x0s_cand, bwd_mean, bwd_std)

    # Compute weights
    w = (sample_fwd - sample_bwd) + (dm_bwd_cond - dm_fwd) * guidance_target + (dm_bwd_uncond - dm_fwd) * (1 - guidance_target)

    denoised_x0s_uncond = net(x0s_cand,  t0s.unsqueeze(0).expand(B, -1), None).clone()
    denoised_x0s_cond = net(x0s_cand,  t0s.unsqueeze(0).expand(B, -1), labels).clone()
    denoised_x0s = guidance * denoised_x0s_cond + (1-guidance) * denoised_x0s_uncond
    denoised_x0s_image = encoder.decode(denoised_x0s)
    reward_x0s = compute_clip_and_imagereward(denoised_x0s_image, prompts, device=device, want_clip=False, want_imr=True)[1].to(device)
    reward_x0s = reward_x0s * r0s

    # Add reward difference to weights
    w += reward_x0s - reward_x1s

    # --- Generate candidate samples x1s_cand from x0s ---
    denoised_x0s_uncond = net(x0s,  t0s.unsqueeze(0).expand(B, -1), None).clone()
    denoised_x0s_cond = net(x0s,  t0s.unsqueeze(0).expand(B, -1), labels).clone()
    denoised_x0s = guidance * denoised_x0s_cond + (1-guidance) * denoised_x0s_uncond
    denoised_x0s_image = encoder.decode(denoised_x0s)
    reward_x0s = compute_clip_and_imagereward(denoised_x0s_image, prompts, device=device, want_clip=False, want_imr=True)[1].to(device)
    reward_x0s = reward_x0s * r0s

    x1s_cand = x0s + fwd_std * torch.randn_like(x0s)

    denoised_cond = net(x1s_cand,  t1s.unsqueeze(0).expand(B, -1), labels).clone()
    score_cond = (-x1s_cand + denoised_cond) / t1s_b**2 
    denoised_uncond = net(x1s_cand,  t1s.unsqueeze(0).expand(B, -1), None).clone()
    score_uncond = (-x1s_cand + denoised_uncond) / t1s_b**2 
    denoised_sampling = guidance * denoised_cond + (1-guidance) * denoised_uncond
    score_sampling = (-x1s_cand + denoised_sampling) / t1s_b**2 

    # Diffusion forward process
    dm_fwd = log_norm_prob(x1s_cand, x0s, fwd_std) 

    # Uncond diffusion backward process
    bwd_mean_uncond = x1s_cand + bwd_std**2 * score_uncond
    dm_bwd_uncond = log_norm_prob(x0s, bwd_mean_uncond, bwd_std)

    # Cond diffusion backward process
    bwd_mean_cond = x1s_cand + bwd_std**2 * score_cond
    dm_bwd_cond = log_norm_prob(x0s, bwd_mean_cond, bwd_std)

    # Sampling forward process
    sample_fwd = log_norm_prob(x1s_cand, x0s, fwd_std) 

    # Sampling backward process
    bwd_mean = x1s_cand + bwd_std**2 * score_sampling 
    sample_bwd = log_norm_prob(x0s, bwd_mean, bwd_std)

    # Compute weights
    w += -(sample_fwd - sample_bwd) - (dm_bwd_cond - dm_fwd) * guidance_target - (dm_bwd_uncond - dm_fwd) * (1 - guidance_target)
    

    denoised_x1s_uncond = net(x1s_cand,  t1s.unsqueeze(0).expand(B, -1), None).clone()
    denoised_x1s_cond = net(x1s_cand,  t1s.unsqueeze(0).expand(B, -1), labels).clone()
    denoised_x1s = guidance * denoised_x1s_cond + (1-guidance) * denoised_x1s_uncond
    denoised_x1s_image = encoder.decode(denoised_x1s)
    reward_x1s = compute_clip_and_imagereward(denoised_x1s_image, prompts, device=device, want_clip=False, want_imr=True)[1].to(device)
    reward_x1s = reward_x1s * r1s

    # Add reward difference to weights
    w += reward_x1s - reward_x0s

    u = torch.rand_like(w).log()
    mask = (u < w)

    print(f"accept: {torch.exp(w)}")

    if mask_last > 0:
        mask[:, -mask_last:] = 1
    mask[:, -1] = 1 # just to set the last to accept
    mask_b = mask.reshape(*mask.shape, 1, 1, 1)
        
    xs[:, start_idx:-1:2] = torch.where(mask_b, x1s_cand, x1s)
    xs[:, start_idx+1::2] = torch.where(mask_b, x0s_cand, x0s)

    return xs, mask


@torch.no_grad()
def main(
    class_idxs=[0],
    T=128,
    num_pt_iterations=512,
    reward_min=1e-10,
    reward_max=1,
    reward_rho=5,
    switch_t_idx=10,
    guidance=1.7,
    guidance_target=1.7,
    init_guidance=1.7,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    local_exploration=False,
    mask_last=0,
    img_net=64
):

    string_class_idxs = "_".join([str(c) for c in class_idxs])
    now = datetime.datetime.now()
    # timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    run_name = f"results/img_net_{img_net}_class_idxs_{string_class_idxs}_local_exploration_{local_exploration}_T_{T}_num_pt_iterations_{num_pt_iterations}_switch_t_idx_{switch_t_idx}_guidance_{guidance}_guidance_target_{guidance_target}"#_{timestamp_str}"

    # --- Setup ---
    net, encoder = get_model(img_net)
    net = BatchedEDM(net).eval()
    encoder = BatchedEncoder(encoder)
    imagenet_labels = get_imagenet_labels()
    shape = (net.img_channels, net.img_resolution, net.img_resolution)    
        
    # --- Generate Labels for the current batch ---
    batch_size = len(class_idxs)
    class_idxs = torch.tensor(class_idxs, device=device)
    assert torch.all((0 <= class_idxs) * (class_idxs < net.label_dim)), f"Class indices must be in [0,{net.label_dim})"
    labels_batch = torch.nn.functional.one_hot(class_idxs, num_classes=net.label_dim)


    prompts_batch = {
        417: ['A blue balloon'],
        723: ['a colorful pinwheel'],
        496: ['a green Christmas stocking'],
        468: ['a yellow cab with dark background'],
        791: ['an empty shopping cart']
        }[class_idxs.item()]
    print(f"Prompts for current batch:\n{prompts_batch}")

    # --- Generate Samples for the current batch ---
    final, accept_probs = apt_sampler(
        net, 
        batch_size, 
        T, 
        labels_batch, 
        shape, 
        num_pt_iterations=num_pt_iterations,
        reward_min=reward_min,
        reward_max=reward_max,
        reward_rho=reward_rho, 
        encoder=encoder,
        prompts=prompts_batch,
        sigma_min=sigma_min, 
        sigma_max=sigma_max, 
        rho=rho, 
        guidance=guidance, 
        guidance_target=guidance_target, 
        init_guidance=init_guidance,
        switch_t_idx=switch_t_idx, 
        local_exploration=local_exploration, 
        mask_last=mask_last)

    print("\nAverage acceptance probabilities:\n")
    print(accept_probs.mean(dim=0))
    print("\nMinimum acceptance probabilities:\n")
    print(accept_probs.min(dim=0).values)

    final = encoder.decode(final.detach())

    save_dict = {
        "final": final.detach().cpu(),
        "accept_probs": accept_probs.detach().cpu(),
        "class_idxs": class_idxs,
        "switch_t_idx": switch_t_idx,
        "guidance": guidance,
        "guidance_target": guidance_target,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "rho": rho,
        "local_exploration": local_exploration,
        "mask_last": mask_last,
        "img_net": img_net,
    }
    save_path = Path(run_name + "_samples.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run APT sampler with configurable settings.")
    parser.add_argument("--T", type=int, default=64, help="Number of temperature steps")
    parser.add_argument("--num_pt_iterations", type=int, default=200, help="Number of PT iterations")
    parser.add_argument("--switch_t_idx", type=int, default=31, help="Switch time index from APT to EDM")
    parser.add_argument("--guidance", type=float, default=1.3, help="Guidance scale")
    parser.add_argument("--guidance_target", type=float, default=1.3, help="Target guidance scale")
    parser.add_argument("--init_guidance", type=float, default=1.3, help="Initial guidance scale")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Rho value")
    parser.add_argument("--local_exploration", action="store_true", help="Enable local exploration")
    parser.add_argument("--mask_last", type=int, default=0, help="Numbers of last indices to mask")
    parser.add_argument("--img_net", type=int, default=64, choices=[64, 512], help="ImageNet size")

    parser.add_argument("--reward_min", type=float, default=1e-10, help="Minimum reward value")
    parser.add_argument("--reward_max", type=float, default=1, help="Maximum reward value")
    parser.add_argument("--reward_rho", type=float, default=5, help="Reward rho value")

    parser.add_argument(
       '--class_idxs',
        type=int,
        nargs='+',
        help='class indices for ImageNet (0-999)',
    )

    args = parser.parse_args()

    print("Arguments:")
    print(args)

    # Pass args to main()
    main(
        T=args.T,
        num_pt_iterations=args.num_pt_iterations,
        switch_t_idx=args.switch_t_idx,
        guidance=args.guidance,
        guidance_target=args.guidance_target,
        init_guidance=args.init_guidance,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        local_exploration=args.local_exploration,
        mask_last=args.mask_last,
        img_net=args.img_net,

        reward_min=args.reward_min,
        reward_max=args.reward_max,
        reward_rho=args.reward_rho,

        class_idxs=args.class_idxs
    )
