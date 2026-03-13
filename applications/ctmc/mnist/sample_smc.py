#!/usr/bin/env python3
"""
Sample from a CFG model trained on MNIST using Sequential Monte Carlo (SMC).

Usage:
    python sample_smc.py --model-path checkpoints/trained_model.pt
    python sample_smc.py --model-path checkpoints/trained_model.pt --num-particles 128 --smc-temperature 2.0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from lib import sampler
from lib.graph import Absorbing, Uniform
from lib.models.unet_tauldr import ConfigTauLDR, get_unet_tauldr
from lib.noise import CosineNoise, GeometricNoise, LogLinearNoise
from lib.sampler import AnalyticPredictor, EulerPredictor


def main():
    parser = argparse.ArgumentParser(description="SMC sampling for MNIST")

    # Model path
    parser.add_argument("--model-path", type=str, required=True)

    # Graph / Noise
    parser.add_argument(
        "--graph", default="absorbing", choices=["absorbing", "uniform"]
    )
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument(
        "--noise", default="cosine", choices=["cosine", "loglinear", "geometric"]
    )
    parser.add_argument("--noise-eps", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=1.0)

    # Predictor
    parser.add_argument("--predictor", default="euler", choices=["euler", "analytic"])

    # Sampling
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")

    # SMC parameters
    parser.add_argument("--proposal-strength", type=float, default=1.0)
    parser.add_argument("--smc-temperature", type=float, default=3.0)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--resample-fraction", type=float, default=0.2)
    parser.add_argument("--resampling-method", default="multinomial")
    parser.add_argument("--sampling-threshold", type=float, default=0.0)

    # Run control
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--compare-cfg", action="store_true", default=True)
    parser.add_argument("--no-compare-cfg", action="store_true")
    parser.add_argument("--cfg-temperature", type=float, default=2.0)
    parser.add_argument("--output-dir", default="./outputs/smc")

    # Model architecture
    parser.add_argument("--scaling-trick", action="store_true", default=True)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--class-embed-dim", type=int, default=32)
    parser.add_argument("--data-dim", type=int, nargs="+", default=[784])
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cfg-train", action="store_true", default=True)
    parser.add_argument("--denoise", action="store_true", default=True)

    args = parser.parse_args()

    # Handle --no-compare-cfg flag
    if args.no_compare_cfg:
        args.compare_cfg = False

    print("=== MNIST SMC Sampling ===")
    print(f"Using {args.num_particles} particles per SMC run")
    print(f"Running {args.num_runs} SMC runs")

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

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
    if args.predictor == "euler":
        predictor = EulerPredictor(graph=graph_module, noise=noise_module)
    else:
        predictor = AnalyticPredictor(graph=graph_module, noise=noise_module)

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

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Create SMC sampler
    smc_sampler = sampler.get_cfg_smc_sampler(
        graph=graph_module,
        noise=noise_module,
        batch_dims=(args.num_particles, *args.data_dim),
        predictor=predictor,
        steps=args.sampling_steps,
        proposal_strength=args.proposal_strength,
        smc_temperature=args.smc_temperature,
        eps=args.eps,
        ess_threshold=args.ess_threshold,
        resample_fraction=args.resample_fraction,
        resampling_method=args.resampling_method,
        sampling_threshold=args.sampling_threshold,
        device=device,
    )

    # Create class labels: 0-9, repeated across runs
    target_class_tensor = torch.arange(0, 10, device=device).repeat(
        args.num_runs // 10 + args.num_runs % 10
    )

    # Run SMC sampling
    all_smc_samples = []
    all_cfg_samples = []

    for run_idx in range(args.num_runs):
        print(f"\nSMC Run {run_idx + 1}/{args.num_runs}")
        print(f"Running {args.num_iterations} SMC iterations per run")

        class_labels = target_class_tensor[run_idx].repeat(args.num_particles)
        current_class = target_class_tensor[run_idx].item()

        for smc_iter in range(args.num_iterations):
            print(f"  SMC Iteration {smc_iter + 1}/{args.num_iterations}")

            with torch.no_grad():
                result = smc_sampler(
                    model,
                    model,
                    class_labels,
                    traces=["ess", "weight", "resampling", "particles"],
                )

            # Collect samples
            samples_np = result.particles.cpu().numpy()
            all_smc_samples.append(samples_np)

            # Save particles grid as image
            particles_reshaped = result.particles.view(-1, 1, 28, 28).float() / 255.0
            nrow = int(np.ceil(np.sqrt(args.num_particles)))
            grid = make_grid(particles_reshaped, nrow=nrow, padding=2, normalize=False)
            grid_path = os.path.join(
                args.output_dir,
                f"smc_run{run_idx}_iter{smc_iter}_class{int(current_class)}.png",
            )
            save_image(grid, grid_path)

            # Print ESS summary
            if result.ess_trace is not None:
                ess_arr = np.array(result.ess_trace)
                print(
                    f"    ESS: min={ess_arr.min():.1f}, "
                    f"max={ess_arr.max():.1f}, "
                    f"final={ess_arr[-1]:.1f}"
                )

            # Print resampling summary
            if result.resampling_trace is not None:
                num_resampled = result.resampling_trace.sum().item()
                print(f"    Resampling events: {int(num_resampled)}")

        # Standard CFG comparison
        if args.compare_cfg:
            print("\n  === Running Standard CFG ===")
            print(f"  Running {args.num_iterations} CFG iterations per run")

            cfg_sampler = sampler.get_pc_sampler_cfg(
                graph=graph_module,
                noise=noise_module,
                batch_dims=(args.num_particles, *args.data_dim),
                predictor=predictor,
                steps=args.sampling_steps,
                eps=args.eps,
                device=device,
            )

            for cfg_iter in range(args.num_iterations):
                print(f"    CFG Iteration {cfg_iter + 1}/{args.num_iterations}")

                with torch.no_grad():
                    cfg_samples, _ = cfg_sampler(
                        model,
                        model,
                        class_labels,
                        cfg_temperature=args.proposal_strength,
                    )

                cfg_samples_np = cfg_samples.cpu().numpy()
                all_cfg_samples.append(cfg_samples_np)

                # Save CFG grid
                cfg_samples_reshaped = cfg_samples.view(-1, 1, 28, 28).float() / 255.0
                nrow = int(np.ceil(np.sqrt(args.num_particles)))
                grid = make_grid(
                    cfg_samples_reshaped, nrow=nrow, padding=2, normalize=False
                )
                grid_path = os.path.join(
                    args.output_dir,
                    f"cfg_run{run_idx}_iter{cfg_iter}_class{int(current_class)}.png",
                )
                save_image(grid, grid_path)

        # Unconditional sampling comparison
        print("\n  === Running Unconditional Sampling ===")

        uncond_sampler = sampler.get_pc_sampler(
            graph=graph_module,
            noise=noise_module,
            batch_dims=(args.num_particles, *args.data_dim),
            predictor=predictor,
            steps=args.sampling_steps,
            eps=args.eps,
            device=device,
        )

        with torch.no_grad():
            uncond_samples, _ = uncond_sampler(model)

        uncond_samples_reshaped = uncond_samples.view(-1, 1, 28, 28).float() / 255.0
        nrow = int(np.ceil(np.sqrt(args.num_particles)))
        grid = make_grid(uncond_samples_reshaped, nrow=nrow, padding=2, normalize=False)
        grid_path = os.path.join(args.output_dir, f"uncond_run{run_idx}.png")
        save_image(grid, grid_path)

    # Save all SMC samples as numpy array
    if len(all_smc_samples) > 0:
        all_samples_array = np.stack(all_smc_samples, axis=0)
        total_samples = all_samples_array.reshape(-1, 784)
        total_samples_reshaped = total_samples.reshape(-1, 1, 28, 28)
        samples_save_path = os.path.join(args.output_dir, "all_smc_samples.npy")
        np.save(samples_save_path, total_samples_reshaped)
        print(f"\nAll SMC samples saved to: {samples_save_path}")
        print(f"Sample array shape: {total_samples_reshaped.shape}")

    # Save all CFG samples as numpy array
    if len(all_cfg_samples) > 0:
        all_cfg_array = np.stack(all_cfg_samples, axis=0)
        total_cfg_samples = all_cfg_array.reshape(-1, 784)
        total_cfg_samples_reshaped = total_cfg_samples.reshape(-1, 1, 28, 28)
        cfg_samples_save_path = os.path.join(args.output_dir, "all_cfg_samples.npy")
        np.save(cfg_samples_save_path, total_cfg_samples_reshaped)
        print(f"All CFG samples saved to: {cfg_samples_save_path}")
        print(f"CFG sample array shape: {total_cfg_samples_reshaped.shape}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
