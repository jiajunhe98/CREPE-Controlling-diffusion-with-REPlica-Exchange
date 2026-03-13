import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from jaxtyping import Float, Int
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from lib import graph, noise, sampler
from lib.data.mnist import DiscreteMNISTDataset
from lib.graph import Absorbing, Uniform
from lib.models import utils as mutils
from lib.models.unet_tauldr import ConfigTauLDR, get_unet_tauldr
from lib.noise import CosineNoise, GeometricNoise, LogLinearNoise
from lib.sampler import EulerPredictor


def compute_score_entropy(
    X0: Int[torch.Tensor, "batch datadim"],
    Xt: Int[torch.Tensor, "batch datadim"],
    log_score: Float[torch.Tensor, "batch datadim vocab"],
    sigma: Float[torch.Tensor, "batch"],
    dsigma: Float[torch.Tensor, "batch"],
    graph: graph.Graph,
) -> Float[torch.Tensor, "batch datadim"]:
    """
    Compute the score entropy with scaling by dsigma.

    Args:
        X0: Original integer samples with shape [batch_size datadim]
        Xt: Noised integer samples with shape [batch_size datadim]
        log_score: Log score with shape [batch_size datadim vocab]
        sigma: Noise level with shape [batch_size]
        dsigma: Noise level derivative with shape [batch_size]
        graph: Graph instance for computing score entropy

    Returns:
        Score entropy scaled by dsigma with shape [batch_size, datadim]
    """
    score_entropy: Float[torch.Tensor, "batch datadim"] = graph.score_entropy(
        log_score, sigma[..., None], Xt, X0
    )
    return score_entropy * dsigma[..., None]


def compute_loss(
    model,
    graph: graph.Graph,
    noise: noise.Noise,
    X0: Int[torch.Tensor, "batch datadim"],
    device: torch.device,
    Y0: Int[torch.Tensor, "batch"] = None,
    cfg_train: bool = False,
    cfg_dropout_prob: float = 0.2,
) -> Float[torch.Tensor, ""]:
    """
    Compute the loss for training the score model on MNIST data.

    Args:
        model: Neural network model
        graph: Graph instance
        noise: Noise scheduler
        X0: Original integer samples with shape [batch_size, datadim]
        device: PyTorch device
        Y0: Class labels with shape [batch_size] (required if cfg_train=True)
        cfg_train: Whether to use classifier-free guidance training
        cfg_dropout_prob: Probability of dropping class conditioning (default 0.2)

    Returns:
        Mean score entropy loss (scalar)
    """
    num_samples = X0.shape[0]
    t = torch.rand(num_samples, device=device)
    sigma, dsigma = noise(t)
    Xt: Int[torch.Tensor, "batch datadim"] = graph.sample_transition(
        X0, sigma[..., None]
    )

    model_fn = mutils.get_score_fn(model, train=True, sampling=False)

    if cfg_train:
        if Y0 is None:
            raise ValueError("Y0 (class labels) must be provided when cfg_train=True")

        # Create mask for classifier-free guidance
        # 1 means use conditioning, 0 means don't use conditioning
        mask = torch.bernoulli(
            torch.full((num_samples,), 1.0 - cfg_dropout_prob, device=device)
        )

        log_score: Float[torch.Tensor, "batch datadim vocab"] = model_fn(
            Xt.float(), sigma, class_labels=Y0, mask=mask
        )
    else:
        # Unconditional training - no class labels or mask needed
        log_score: Float[torch.Tensor, "batch datadim vocab"] = model_fn(
            Xt.float(), sigma
        )

    score_entropy = compute_score_entropy(X0, Xt, log_score, sigma, dsigma, graph)
    return score_entropy.mean()


@dataclass
class TrainStepOutput:
    stats: dict


def train_step(
    batch,
    model,
    graph_module,
    noise_module,
    optimizer,
    device,
    cfg_train,
    cfg_dropout_prob,
    max_grad_norm=None,
) -> TrainStepOutput:
    if cfg_train:
        # Conditional training - expect (data, labels) pairs
        X0: Int[torch.Tensor, "batch datadim"]
        Y0: Int[torch.Tensor, "batch"]
        X0, Y0 = batch
        X0 = X0.int().to(device)
        Y0 = Y0.long().to(device)
    else:
        # Unconditional training - expect only data
        X0: Int[torch.Tensor, "batch datadim"] = batch.int().to(device)
        Y0 = None

    optimizer.zero_grad()

    loss = compute_loss(
        model, graph_module, noise_module, X0, device, Y0, cfg_train, cfg_dropout_prob
    )
    loss.backward()

    # Clip gradients if max_grad_norm is specified
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    return TrainStepOutput(stats={"loss": loss.item()})


def main():
    parser = argparse.ArgumentParser(description="Train SEDD on MNIST")

    # Graph/Noise
    parser.add_argument("--graph", default="uniform", choices=["uniform", "absorbing"])
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument(
        "--noise", default="geometric", choices=["geometric", "loglinear", "cosine"]
    )
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--noise-eps", type=float, default=1e-3)

    # Training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--scaling-trick", action="store_true", default=True)
    parser.add_argument("--cfg-train", action="store_true", default=True)
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.2)
    parser.add_argument("--cfg-temperature", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--data-dim", type=int, nargs="+", default=[784])

    # UNet architecture
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--class-embed-dim", type=int, default=32)

    args = parser.parse_args()

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build graph
    graph_module = (
        Absorbing(dim=args.vocab_size)
        if args.graph == "absorbing"
        else Uniform(dim=args.vocab_size)
    )

    # Build noise schedule
    NOISE_MAP = {
        "geometric": lambda: GeometricNoise(
            sigma_min=args.sigma_min, sigma_max=args.sigma_max
        ),
        "loglinear": lambda: LogLinearNoise(eps=args.noise_eps),
        "cosine": lambda: CosineNoise(eps=args.noise_eps),
    }
    noise_module = NOISE_MAP[args.noise]()

    # Build predictor
    predictor = EulerPredictor(graph=graph_module, noise=noise_module)

    # Build model
    dataS = args.vocab_size + 1 if args.graph == "absorbing" else args.vocab_size
    model_cfg = ConfigTauLDR(
        dataS=dataS,
        num_scales=args.num_scales,
        num_res_blocks=args.num_res_blocks,
        ch_mult=args.ch_mult,
        scaling_trick=args.scaling_trick,
        class_embed_dim=args.class_embed_dim,
        ch=args.ch,
    )
    model = get_unet_tauldr(model_cfg, cfg_train=args.cfg_train).to(device)

    # Build optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    # Build dataset and dataloader
    dataset = DiscreteMNISTDataset(
        root="~/data/mnist",
        train=True,
        download=True,
        with_targets=args.cfg_train,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create sampler
    if args.cfg_train:
        pc_sampler = sampler.get_pc_sampler_cfg(
            graph=graph_module,
            noise=noise_module,
            batch_dims=(args.num_samples, *args.data_dim),
            predictor=predictor,
            steps=args.sampling_steps,
            device=device,
        )
    else:
        pc_sampler = sampler.get_pc_sampler(
            graph=graph_module,
            noise=noise_module,
            batch_dims=(args.num_samples, *args.data_dim),
            predictor=predictor,
            steps=args.sampling_steps,
            device=device,
        )

    # Create EMA model
    ema_model = None
    if args.ema_decay is not None:
        ema_model = mutils.create_ema_model(model, args.ema_decay, device)

    # Training loop
    training_done = False
    global_training_step = 0
    while not training_done:
        for batch in dataloader:
            train_step_output = train_step(
                batch,
                model,
                graph_module,
                noise_module,
                optimizer,
                device,
                args.cfg_train,
                args.cfg_dropout_prob,
                args.max_grad_norm,
            )

            print(
                f"Step {global_training_step}: loss={train_step_output.stats['loss']:.6f}"
            )

            # Update EMA model if it exists
            if ema_model is not None:
                ema_model.update_parameters(model)

            global_training_step += 1

            training_done = global_training_step >= args.num_steps
            if training_done:
                break

    # Sample from model (use EMA model if available)
    inference_model = ema_model if ema_model is not None else model
    with torch.no_grad():
        if args.cfg_train:
            # CFG sampling: sample one image from each class (0-9)
            num_classes = 10
            class_labels = torch.arange(num_classes, device=device).repeat(
                args.num_samples // num_classes + 1
            )[: args.num_samples]

            # For CFG, use the same model for both conditional and unconditional
            samples, initial_samples = pc_sampler(
                inference_model,
                inference_model,
                class_labels,
                args.cfg_temperature,
            )
        else:
            # Unconditional sampling
            samples, initial_samples = pc_sampler(inference_model)

    # Convert samples to images
    sample_images = samples.cpu().reshape(-1, 1, 28, 28) / 255.0  # Normalize to [0,1]

    # Create grid of samples and save
    grid = make_grid(sample_images, nrow=8, padding=2, normalize=False)

    os.makedirs(args.save_dir, exist_ok=True)
    save_image(grid, os.path.join(args.save_dir, "generated_samples.png"))
    print(f"Saved generated samples to {args.save_dir}/generated_samples.png")

    # Save trained model
    model_path = os.path.join(args.save_dir, "trained_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Save EMA model if it exists
    if ema_model is not None:
        ema_model_path = os.path.join(args.save_dir, "ema_model.pt")
        torch.save(ema_model.state_dict(), ema_model_path)
        print(f"Saved EMA model to {ema_model_path}")


if __name__ == "__main__":
    main()
