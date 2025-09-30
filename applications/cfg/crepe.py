import argparse
import pickle, torch, PIL.Image
from pathlib import Path
import dnnlib
import numpy as np
from torch_utils.misc import tile_images
from tqdm import tqdm
import json
import clip
import ImageReward as RM

torch.manual_seed(1)
device = torch.device('cuda')


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


class BatchedEDM(torch.nn.Module):
    """
    A wrapper for the EDM model that automatically handles multiple batch dimensions
    and large batch sizes with automatic chunking.

    This allows you to pass inputs with shapes like [B1, B2, C, H, W]
    without needing to manually flatten and reshape them.
    """
    def __init__(self, model, max_batch_size=2048):
        super().__init__()
        self.model = model
        self.max_batch_size = max_batch_size

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
        # Store the original shape and number of batch dimensions
        original_shape = x.shape
        num_batch_dims = x.ndim - 3 # The last 3 dims are (C, H, W)

        # If there's only one batch dimension, check if we need chunking
        if num_batch_dims <= 1:
            if x.shape[0] > self.max_batch_size:
                return self._chunked_forward(x, t, labels, **kwargs)
            else:
                return self.model(x, t, labels, **kwargs)

        # Flatten the batch and time dimensions
        # Reshape x from [B1, B2, C, H, W] to [B, C, H, W]
        flattened_x = x.reshape(-1, *original_shape[num_batch_dims:])
        # Reshape t from [B1, B2] to [B]
        flattened_t = t.reshape(-1)

        flattened_labels = None
        if labels is not None:
            # Expand labels from [B1, D] to [B, D]
            flattened_labels = labels.unsqueeze(1).expand(-1, x.shape[1], -1).reshape(-1, labels.shape[-1])

        # Check if flattened tensor needs chunking
        if flattened_x.shape[0] > self.max_batch_size:
            output = self._chunked_forward(flattened_x, flattened_t, flattened_labels, **kwargs)
        else:
            output = self.model(flattened_x, flattened_t, flattened_labels, **kwargs)

        # Reshape the output back to the original batch structure
        reshaped_output = output.reshape(original_shape)

        return reshaped_output

    def _chunked_forward(self, x, t, labels=None, **kwargs):
        batch_size = x.shape[0]
        chunk_size = min(self.max_batch_size, batch_size)

        # Pre-allocate output tensor
        device = x.device
        dtype = x.dtype

        # Run first chunk to get output shape
        end_idx = chunk_size
        chunk_x = x[:end_idx]
        chunk_t = t[:end_idx] if t.numel() > 1 else t
        chunk_labels = labels[:end_idx] if labels is not None else None

        first_output = self.model(chunk_x, chunk_t, chunk_labels, **kwargs)
        output_shape = (batch_size, *first_output.shape[1:])
        output = torch.zeros(output_shape, device=device, dtype=dtype)
        output[:end_idx] = first_output

        # Process remaining chunks
        for i in range(chunk_size, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_x = x[i:end_idx]
            chunk_t = t[i:end_idx] if t.numel() > 1 else t
            chunk_labels = labels[i:end_idx] if labels is not None else None

            output[i:end_idx] = self.model(chunk_x, chunk_t, chunk_labels, **kwargs)

        return output


class BatchedEncoder:
    """
    A wrapper for an Encoder that automatically handles multiple batch dimensions
    and large batch sizes with automatic chunking.
    """
    def __init__(self, encoder, max_batch_size=2048):
        self.encoder = encoder
        self.max_batch_size = max_batch_size

    def _apply_with_batching(self, func, x):
        original_shape = x.shape
        num_batch_dims = x.ndim - 3

        if num_batch_dims <= 1:
            if x.shape[0] > self.max_batch_size:
                return self._chunked_apply(func, x)
            else:
                return func(x)

        flattened_x = x.reshape(-1, *original_shape[num_batch_dims:])

        if flattened_x.shape[0] > self.max_batch_size:
            output = self._chunked_apply(func, flattened_x)
        else:
            output = func(flattened_x)

        reshaped_output = output.reshape(original_shape)
        return reshaped_output

    def _chunked_apply(self, func, x):
        batch_size = x.shape[0]
        chunk_size = min(self.max_batch_size, batch_size)

        end_idx = chunk_size
        first_output = func(x[:end_idx])

        device = x.device
        output_dtype = first_output.dtype
        output_shape = (batch_size, *first_output.shape[1:])
        output = torch.zeros(output_shape, device=device, dtype=output_dtype)
        output[:end_idx] = first_output

        for i in range(chunk_size, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            output[i:end_idx] = func(x[i:end_idx])

        return output

    def decode(self, latents):
        return self._apply_with_batching(self.encoder.decode, latents)

    def encode_latents(self, latents):
        return self._apply_with_batching(self.encoder.encode_latents, latents)

    def encode_pixels(self, pixels):
        return self._apply_with_batching(self.encoder.encode_pixels, pixels)

    def encode(self, pixels):
        return self._apply_with_batching(self.encoder.encode, pixels)


@torch.no_grad()
def edm_sampler(net, initial, t_steps, guidance, labels=None,):
    def denoise(x, t):
        Dx = net(x, t, labels).clone()
        if guidance == 1.0 or labels is None:
            return Dx
        ref_Dx = net(x, t, None)
        return ref_Dx.lerp(Dx, guidance)

    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x = initial
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        score = (-x + denoise(x, t_cur)) / t_cur**2
        x = x + t_cur * (t_cur -  t_next / 2) * score + torch.sqrt(t_next**2 - t_next**2 / 4) * torch.randn_like(x)
    return x


def log_norm_prob(x, mu, std):
    std = std.expand(x.shape)
    return -0.5 * ((x - mu) / std).pow(2).sum(dim=(-3, -2,-1)) - (std.log() + np.log(2*np.pi)/2).sum(dim=(-3, -2,-1))


@torch.no_grad()
def edm_sampler_for_initialization(net, noise, labels, t_steps, guidance=1.7):
    initial = noise * t_steps[0]
    x = edm_sampler(net, initial, t_steps, guidance, labels=labels)
    x = x + t_steps.reshape(1, -1, 1, 1, 1) * torch.randn_like(x)
    return x


@torch.no_grad()
def crepe_iteration(xs, temp_schedule, labels, net, start_idx, sigma_max, guidance, guidance_target, local_exploration, mask_last):
    """Single CREPE iteration"""
    B, T = xs.shape[0], xs.shape[1]

    # Batched Local Exploration
    if local_exploration:
        denoised_cond = net(xs,  temp_schedule.unsqueeze(0).expand(B, -1), labels).clone()
        denoised_uncond = net(xs,  temp_schedule.unsqueeze(0).expand(B, -1), None).clone()
        denoised_sampling = guidance_target * denoised_cond + (1-guidance_target) * denoised_uncond
        score_sampling = (-xs + denoised_sampling) / temp_schedule.reshape(1, -1, 1, 1, 1)**2
        lang_std = torch.sqrt(2 * temp_schedule[:-1] * (temp_schedule[:-1] - temp_schedule[1:])).reshape(1, -1, 1, 1, 1)
        xs[:, :-1] = xs[:, :-1] + 0.5 * lang_std**2 * score_sampling[:, :-1] + lang_std * torch.randn_like(xs[:, :-1]) # No langevin at lowest time

    # Resample highest time
    xs[:, 0] = sigma_max * torch.randn_like(xs[:, 0])

    # CREPE
    x1s = xs[:, start_idx:-1:2]
    x0s = xs[:, start_idx+1::2]
    idx1s = torch.arange(T)[start_idx:-1:2]
    idx0s = torch.arange(T)[start_idx+1::2]
    t1s = temp_schedule[idx1s]
    t0s = temp_schedule[idx0s]
    dt = (t1s - t0s).abs()

    t1s_b = t1s.reshape(1, -1, 1, 1, 1)
    t0s_b = t0s.reshape(1, -1, 1, 1, 1)
    dt_b = dt.reshape(1, -1, 1, 1, 1)

    # Denoising std
    bwd_std = torch.sqrt(2 * t1s_b * dt_b)

    # Noising std
    fwd_std = torch.sqrt(2 * t0s_b * dt_b)

    # Generate candidate samples x0s_cand from x1s
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

    w = (sample_fwd - sample_bwd) + (dm_bwd_cond - dm_fwd) * guidance_target + (dm_bwd_uncond - dm_fwd) * (1 - guidance_target)

    # Generate candidate samples x1s_cand from x0s
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

    w += -(sample_fwd - sample_bwd) - (dm_bwd_cond - dm_fwd) * guidance_target - (dm_bwd_uncond - dm_fwd) * (1 - guidance_target)

    u = torch.rand_like(w).log()
    mask = (u < w)

    if mask_last > 0:
        mask[:, -mask_last:] = 1
    mask_b = mask.reshape(*mask.shape, 1, 1, 1)

    xs[:, start_idx:-1:2] = torch.where(mask_b, x1s_cand, x1s)
    xs[:, start_idx+1::2] = torch.where(mask_b, x0s_cand, x0s)

    return xs, mask


@torch.no_grad()
def crepe_sampler(net, B, labels, shape, num_pt_iterations, T=128, sigma_min=0.002, sigma_max=80.0, rho=7.0, guidance=1.7, guidance_target=1.7, switch_t_idx=0, local_exploration=False, mask_last=0, return_all=False, **kwargs):
    """Full CREPE sampler implementation"""
    t_schedule = (sigma_max**(1/rho) + torch.linspace(0, 1, T, device=device) * (sigma_min**(1/rho) - sigma_max**(1/rho))).pow(rho)

    initial_noise = torch.randn(B, 1, *shape, device=device)
    print("Initializing chains.")
    xs = edm_sampler_for_initialization(net, initial_noise, labels, t_schedule, guidance=guidance)
    print("Chains initialized.")

    print(f"Switching from CREPE to SDE at time {t_schedule[T-1-switch_t_idx]}.")
    xs = xs[:,:T-switch_t_idx]

    results = torch.empty(B, num_pt_iterations * 2, *shape, device=device)
    accept_probs = torch.zeros(B, T-1-switch_t_idx, device=device)

    if return_all:
        full_trajectory = torch.empty(B, num_pt_iterations * 2, T-switch_t_idx, *shape, device=device)
    else:
        full_trajectory = None

    print("Starting CREPE.")
    sample_idx = 0
    for _ in tqdm(range(num_pt_iterations)):
        for start_idx in [0, 1]:
            xs, mask = crepe_iteration(xs, t_schedule[:T-switch_t_idx], labels, net, start_idx, sigma_max, guidance, guidance_target, local_exploration, mask_last)
            results[:, sample_idx] = xs[:, -1].clone()

            if return_all:
                full_trajectory[:, sample_idx] = xs.clone()

            sample_idx += 1
            accept_probs[:, start_idx::2] += mask

    accept_probs /= num_pt_iterations

    # Final SDE integration
    print("\nStarting final SDE integration.")

    if labels is not None:
        labels_expanded = labels.repeat_interleave(results.shape[1], dim=0)
    else:
        labels_expanded = None

    results_flat = results.reshape(-1, *results.shape[2:])  # [B*num_samples, C, H, W]
    final_results_flat = edm_sampler(net, results_flat, t_schedule[T-switch_t_idx-1:], guidance, labels=labels_expanded)
    final = final_results_flat.reshape(B, results.shape[1], *final_results_flat.shape[1:])

    if return_all:
        return final, accept_probs, full_trajectory
    else:
        return final, accept_probs


@torch.no_grad()
def crepe_batched_sampler(net, num_particles, num_classes, shape, T=128, sigma_min=0.002, sigma_max=80.0, rho=7.0, guidance=1.7, guidance_target=1.7, switch_t_idx=0, local_exploration=False, mask_last=0, **kwargs):
    """
    Run CREPE for multiple classes in parallel

    Args:
        net: diffusion model
        num_particles: target number of particles per class
        num_classes: number of classes to sample
        shape: (C, H, W)

    Returns:
        all_particles: [num_classes, num_particles, C, H, W]
        class_labels: [num_classes] class indices used
        avg_acceptance: average acceptance rate
    """
    # Sample random classes
    class_idxs = torch.randint(net.label_dim, size=(num_classes,), device=device)
    labels_batch = torch.nn.functional.one_hot(class_idxs, num_classes=net.label_dim).float()
    num_pt_iterations = num_particles // 2

    final_samples, accept_probs = crepe_sampler(
        net, num_classes, labels_batch, shape, num_pt_iterations=num_pt_iterations,
        T=T, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        guidance=guidance, guidance_target=guidance_target,
        switch_t_idx=switch_t_idx, local_exploration=local_exploration, mask_last=mask_last,
        **kwargs
    )

    # Take first num_particles from each class
    if final_samples.shape[1] >= num_particles:
        final_samples = final_samples[:, :num_particles]
    else:
        # If we don't have enough, repeat some samples
        repeats = (num_particles + final_samples.shape[1] - 1) // final_samples.shape[1]
        final_samples = final_samples.repeat(1, repeats, 1, 1, 1)[:, :num_particles]

    avg_acceptance = accept_probs.mean().item() if accept_probs is not None else 0.0

    return final_samples, class_idxs.cpu().numpy(), avg_acceptance


def run_experiment(num_particles, num_classes, total_samples, img_net, output_dir, **kwargs):
    """
    Main experiment runner

    Args:
        num_particles: particles per batch
        num_classes: classes per batch
        total_samples: target total number of samples
        img_net: 64 or 512
        output_dir: base output directory
    """
    # Setup
    method = "crepe"
    net, encoder = get_model(img_net)
    net = BatchedEDM(net).eval()
    encoder = BatchedEncoder(encoder)
    shape = (net.img_channels, net.img_resolution, net.img_resolution)

    # Create experiment directory
    exp_name = f"{method}_particles_{num_particles}_img_{img_net}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {method.upper()} experiment: {num_particles} particles, {total_samples} total samples")
    print(f"Output directory: {exp_dir}")

    # Calculate number of batches needed
    samples_per_batch = num_particles * num_classes
    num_batches = (total_samples + samples_per_batch - 1) // samples_per_batch

    # Save experiment config
    config = {
        'method': method,
        'num_particles': num_particles,
        'num_classes': num_classes,
        'total_samples': total_samples,
        'img_net': img_net,
        'num_batches': num_batches,
        'samples_per_batch': samples_per_batch,
        **kwargs
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run batches and save samples to disk
    total_saved = 0
    all_class_labels = []
    all_metrics = []

    for batch_idx in tqdm(range(num_batches), desc="Experiment Batches"):
        print(f"\nBatch {batch_idx + 1}/{num_batches}")

        # Run sampler
        particles, class_labels, metric = crepe_batched_sampler(
            net, num_particles, num_classes, shape, **kwargs
        )
        all_metrics.append(float(metric))  # Acceptance rate

        # Decode particles to images
        print("Decoding images.")
        particles_flat = particles.reshape(-1, *particles.shape[2:])  # [total_samples_this_batch, C, H, W]
        images_uint8 = encoder.decode(particles_flat)

        # Save images to disk
        for i, img in enumerate(images_uint8):
            sample_idx = total_saved + i
            img_path = exp_dir / f"sample_{sample_idx:06d}.png"

            # Convert to PIL and save
            img_np = img.permute(1, 2, 0).cpu().numpy()
            try:
                PIL.Image.fromarray(img_np).save(img_path)
            except FileNotFoundError as e:
                print(f"ERROR: Could not save to {img_path}")
                raise e

        # Save class labels for this batch - expand to match all samples
        batch_labels_expanded = []
        for class_idx in class_labels:
            # Need to repeat each class label num_particles times
            batch_labels_expanded.extend([int(class_idx)] * num_particles)
        all_class_labels.extend(batch_labels_expanded)

        total_saved += len(images_uint8)

        # Clean up GPU memory
        del particles, images_uint8
        torch.cuda.empty_cache()

        print(f"Saved {len(particles_flat)} samples. Total: {total_saved}/{total_samples}")

        if total_saved >= total_samples:
            break

    # Save final metadata
    metadata = {
        'total_samples_saved': int(total_saved),
        'class_labels': all_class_labels[:total_saved],
        'average_metric': float(np.mean(all_metrics)),
        'metric_name': 'acceptance_rate'
    }

    with open(exp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{method.upper()} experiment completed!")
    print(f"Saved {total_saved} samples to {exp_dir}")
    print(f"Average {metadata['metric_name']}: {metadata['average_metric']:.4f}")

    del net, encoder
    torch.cuda.empty_cache()

    return exp_dir


def generate_sample_grid(exp_dir, grid_size=8, max_samples=64):
    """Generate a grid of sample images for visualization"""
    exp_dir = Path(exp_dir)
    print(f"Generating sample grid for: {exp_dir}")

    with open(exp_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    total_samples = min(metadata['total_samples_saved'], max_samples)

    sample_indices = np.random.choice(metadata['total_samples_saved'], total_samples, replace=False)
    images = []

    for idx in sample_indices:
        img_path = exp_dir / f"sample_{idx:06d}.png"
        img = PIL.Image.open(img_path)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        images.append(img_tensor)

    # Create grid
    images_tensor = torch.stack(images)
    grid = tile_images(images_tensor, w=grid_size, h=grid_size)
    grid_np = grid.permute(1, 2, 0).numpy()

    # Save grid
    grid_path = exp_dir / "sample_grid.png"
    PIL.Image.fromarray(grid_np).save(grid_path)

    print(f"Sample grid saved to: {grid_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CREPE experiments")

    parser.add_argument("--particles", nargs='+', type=int, default=[8, 16, 32, 64, 128, 256], help="Number of particles to test")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes per batch")
    parser.add_argument("--total_samples", type=int, default=8000, help="Target total number of samples")
    parser.add_argument("--img_net", type=int, default=64, choices=[64, 512], help="ImageNet resolution")
    parser.add_argument("--output_dir", type=str, default="edm2/outputs", help="Output directory")

    parser.add_argument("--T", type=int, default=128, help="Number of time steps")
    parser.add_argument("--guidance", type=float, default=1.7, help="Guidance scale")
    parser.add_argument("--guidance_target", type=float, default=1.7, help="Target guidance")
    parser.add_argument("--switch_t_idx", type=int, default=30, help="Switch point for final sampling")
    parser.add_argument("--local_exploration", action="store_true", help="Enable local exploration in CREPE")
    parser.add_argument("--mask_last", type=int, default=2, help="Mask last steps in CREPE")

    parser.add_argument("--exp_dir", type=str, help="Experiment directory")
    parser.add_argument("--reference_features", type=str, help="Path to reference features for FID computation")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to use for metrics computation")

    args = parser.parse_args()

    for num_particles in args.particles:
        exp_dir = run_experiment(
            num_particles=num_particles,
            num_classes=args.num_classes,
            total_samples=args.total_samples,
            img_net=args.img_net,
            output_dir=args.output_dir,
            T=args.T,
            guidance=args.guidance,
            guidance_target=args.guidance_target,
            switch_t_idx=args.switch_t_idx,
            local_exploration=args.local_exploration,
            mask_last=args.mask_last
        )

        generate_sample_grid(exp_dir)
