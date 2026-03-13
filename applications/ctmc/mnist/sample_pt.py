#!/usr/bin/env python3
"""
Sample from a CFG model trained on MNIST using Parallel Tempering (PT).

Usage:
    python sample_pt.py --model-path checkpoints/trained_model.pt
    python sample_pt.py --model-path checkpoints/trained_model.pt --num-steps 2000 --burn-in-steps 500
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


@torch.no_grad()
def generate_initial_path(
    graph,
    noise,
    predictor,
    model,
    cond,
    time_steps,
    device,
    data_dim,
    cfg_temperature,
    eps,
):
    """Generate initial path using CFG proposal."""
    cfg_sampler = sampler.get_pc_sampler_cfg(
        graph=graph,
        noise=noise,
        batch_dims=(cond.shape[0], *data_dim),
        predictor=predictor,
        steps=len(time_steps) - 1,
        eps=eps,
        device=device,
        sampling_schedule="linear",
        return_path=True,
    )

    final_samples, initial_samples, path_tensor = cfg_sampler(
        model, model, cond, cfg_temperature=cfg_temperature
    )

    return path_tensor.squeeze(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Tempering sampling for MNIST"
    )

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

    # PT parameters
    parser.add_argument("--path-length", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=1324)
    parser.add_argument("--burn-in-steps", type=int, default=300)
    parser.add_argument("--cfg-temperature", type=float, default=2.0)
    parser.add_argument("--smc-temperature", type=float, default=1.2)
    parser.add_argument("--method", default="prob", choices=["prob", "score"])
    parser.add_argument("--num-local-steps", type=int, default=0)
    parser.add_argument("--force-swap-at-one", action="store_true", default=False)
    parser.add_argument("--keep-burn-in", action="store_true", default=True)
    parser.add_argument("--proposal-strength", type=float, default=1.0)

    # Run control
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--compare-cfg", action="store_true", default=True)
    parser.add_argument("--no-compare-cfg", action="store_true")
    parser.add_argument("--compare-unconditional", action="store_true", default=True)
    parser.add_argument("--no-compare-unconditional", action="store_true")

    # Sampling
    parser.add_argument("--sampling-schedule", default="linear")
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="./outputs/pt")

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

    # Handle negation flags
    if args.no_compare_cfg:
        args.compare_cfg = False
    if args.no_compare_unconditional:
        args.compare_unconditional = False

    print("=== MNIST PT Sampling ===")
    print(f"Path length: {args.path_length}")
    print(f"PT steps: {args.num_steps}, Burn-in: {args.burn_in_steps}")
    print(f"Running {args.num_runs} PT runs")

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

    if args.sampling_schedule != "linear":
        raise ValueError("Only linear schedule is supported for PT sampling.")

    # Create time discretization for paths
    time_steps = sampler.sampling_schedule_grid(
        args.sampling_schedule,
        args.path_length,
        args.eps,
        device,
    )
    step_size = (1 - args.eps) / args.path_length

    # Create PT sampler
    pt_sampler = sampler.get_pt_sampler(
        graph=graph_module,
        noise=noise_module,
        time_steps=time_steps,
        step_size=step_size,
        smc_temperature=args.smc_temperature,
        num_steps=args.num_steps,
        burn_in_steps=args.burn_in_steps,
        cfg_temperature=args.cfg_temperature,
        store_k_paths=50,
        store_every_n_steps=10,
        num_local_steps=args.num_local_steps,
        force_swap_at_one=args.force_swap_at_one,
        method=args.method,
        keep_burn_in=args.keep_burn_in,
    )

    # Create class labels from 0 to 9, repeated across runs
    target_class_tensor = torch.arange(0, 10, device=device).repeat(
        args.num_runs // 10 + args.num_runs % 10
    )

    # Collect all PT samples for numpy array export
    all_pt_samples = []
    all_burn_in_samples = []

    # Run PT sampling multiple times
    for run_idx in range(args.num_runs):
        print(f"\nPT Run {run_idx + 1}/{args.num_runs}")

        # Create conditioning
        class_labels = target_class_tensor[run_idx].repeat(1)

        # Generate initial path
        print("  Generating initial path...")
        initial_path = generate_initial_path(
            graph_module,
            noise_module,
            predictor,
            model,
            class_labels,
            time_steps,
            device,
            args.data_dim,
            args.cfg_temperature,
            args.eps,
        )

        print(f"  Initial path shape: {initial_path.shape}")
        assert initial_path.shape[0] == time_steps.shape[0], (
            f"initial_path: {initial_path.shape}, time_steps: {time_steps.shape}"
        )

        # Save initial path evolution as grid image
        initial_path_imgs = initial_path.view(-1, 1, 28, 28).float()
        step_indices = list(
            range(0, len(initial_path_imgs), max(1, len(initial_path_imgs) // 100))
        )
        if step_indices[-1] != len(initial_path_imgs) - 1:
            step_indices.append(len(initial_path_imgs) - 1)

        evolution_imgs = initial_path_imgs[step_indices]
        nrow = len(step_indices)
        evolution_grid = make_grid(
            evolution_imgs, nrow=nrow, padding=2, scale_each=True
        )
        current_class = target_class_tensor[run_idx].item()
        save_image(
            evolution_grid,
            os.path.join(
                args.output_dir,
                f"initial_path_run{run_idx}_class{int(current_class)}.png",
            ),
        )

        # Run PT sampling
        print("  Running parallel tempering...")
        with torch.no_grad():
            result = pt_sampler(
                model,
                model,
                class_labels,
                initial_path,
                traces=["accept_mask", "log_weights_reverse", "log_weights_forward"],
            )

        # Log acceptance statistics from traces
        if result.accept_count_per_timestep is not None:
            accept_rates_per_timestep = (
                result.accept_count_per_timestep.float() / args.num_steps
            )
            mean_acceptance_rate = accept_rates_per_timestep.mean().item()
            print(f"  Mean acceptance rate: {mean_acceptance_rate:.3f}")

        # Log weight statistics per timestep
        if (
            result.log_weights_reverse_per_timestep is not None
            and result.log_weights_forward_per_timestep is not None
        ):
            mean_log_weights_reverse = (
                result.log_weights_reverse_per_timestep
                / result.weights_count_per_timestep.float()
            )
            mean_log_weights_forward = (
                result.log_weights_forward_per_timestep
                / result.weights_count_per_timestep.float()
            )
            mean_weights_reverse = torch.exp(mean_log_weights_reverse)
            mean_weights_forward = torch.exp(mean_log_weights_forward)
            mean_weights_product = mean_weights_reverse * mean_weights_forward
            print(
                f"  Weight product: min={mean_weights_product.min().item():.4f}, "
                f"max={mean_weights_product.max().item():.4f}, "
                f"mean={mean_weights_product.mean().item():.4f}"
            )

        # Handle burn-in samples
        if result.burn_in_samples is not None and len(result.burn_in_samples) > 0:
            print(f"  Collected {len(result.burn_in_samples)} samples during burn-in")
            burn_in_samples_tensor = torch.stack(result.burn_in_samples)
            burn_in_samples_reshaped = (
                burn_in_samples_tensor.view(-1, 1, 28, 28).float() / 255.0
            )

            burn_in_samples_np = burn_in_samples_tensor.cpu().numpy()
            all_pt_samples.append(burn_in_samples_np)
            all_burn_in_samples.append(burn_in_samples_np)

            nrow = min(int(np.ceil(np.sqrt(len(result.burn_in_samples)))), 8)
            burn_in_grid = make_grid(
                burn_in_samples_reshaped, nrow=nrow, padding=2, scale_each=True
            )
            save_image(
                burn_in_grid,
                os.path.join(
                    args.output_dir,
                    f"pt_burn_in_run{run_idx}_class{int(current_class)}.png",
                ),
            )

        # Handle post-burn-in samples
        print(f"  Collected {len(result.samples)} samples after burn-in")
        if len(result.samples) > 0:
            samples_tensor = torch.stack(result.samples)
            samples_reshaped = samples_tensor.view(-1, 1, 28, 28).float() / 255.0

            samples_np = samples_tensor.cpu().numpy()
            all_pt_samples.append(samples_np)

            nrow = min(int(np.ceil(np.sqrt(len(result.samples)))), 8)
            grid = make_grid(samples_reshaped, nrow=nrow, padding=2, scale_each=True)
            save_image(
                grid,
                os.path.join(
                    args.output_dir,
                    f"pt_samples_run{run_idx}_class{int(current_class)}.png",
                ),
            )

        # Save final path evolution as grid image
        final_path_imgs = result.final_path.view(-1, 1, 28, 28).float()
        step_indices = list(
            range(0, len(final_path_imgs), max(1, len(final_path_imgs) // 100))
        )
        if step_indices[-1] != len(final_path_imgs) - 1:
            step_indices.append(len(final_path_imgs) - 1)

        evolution_imgs = final_path_imgs[step_indices]
        nrow = len(step_indices)
        evolution_grid = make_grid(
            evolution_imgs, nrow=nrow, padding=2, scale_each=True
        )
        save_image(
            evolution_grid,
            os.path.join(
                args.output_dir,
                f"final_path_run{run_idx}_class{int(current_class)}.png",
            ),
        )

        # Save stored k-paths evolution if available
        if result.k_paths_storage is not None:
            print(f"  Saving {result.k_paths_storage.shape[0]} stored paths as grid...")

            all_path_evolutions = []
            for k_idx in range(result.k_paths_storage.shape[0]):
                path_imgs = result.k_paths_storage[k_idx].view(-1, 1, 28, 28).float()
                step_indices = list(
                    range(0, len(path_imgs), max(1, len(path_imgs) // 100))
                )
                if step_indices[-1] != len(path_imgs) - 1:
                    step_indices.append(len(path_imgs) - 1)
                evolution_imgs = path_imgs[step_indices]
                all_path_evolutions.append(evolution_imgs)

            # Grid: each row is one path evolution
            all_evolutions_tensor = torch.cat(all_path_evolutions, dim=0)
            num_steps_per_path = len(step_indices)
            k_paths_grid = make_grid(
                all_evolutions_tensor,
                nrow=num_steps_per_path,
                padding=2,
                scale_each=True,
            )
            save_image(
                k_paths_grid,
                os.path.join(
                    args.output_dir,
                    f"k_paths_run{run_idx}_class{int(current_class)}.png",
                ),
            )

            # Transposed grid: k-paths on x-axis, time steps on y-axis
            num_k_paths = result.k_paths_storage.shape[0]
            transposed_evolutions = []
            for step_idx in range(num_steps_per_path):
                for k_idx in range(num_k_paths):
                    transposed_evolutions.append(all_path_evolutions[k_idx][step_idx])

            transposed_tensor = torch.stack(transposed_evolutions, dim=0)
            k_paths_transposed_grid = make_grid(
                transposed_tensor,
                nrow=num_k_paths,
                padding=2,
                scale_each=True,
            )
            save_image(
                k_paths_transposed_grid,
                os.path.join(
                    args.output_dir,
                    f"k_paths_transposed_run{run_idx}_class{int(current_class)}.png",
                ),
            )

            # Save k-paths as numpy array with full path length
            k_paths_full = (
                result.k_paths_storage.view(
                    result.k_paths_storage.shape[0], -1, 1, 28, 28
                )
                .cpu()
                .numpy()
            )
            k_paths_save_path = os.path.join(
                args.output_dir,
                f"k_paths_full_run{run_idx}_class{int(current_class)}.npy",
            )
            np.save(k_paths_save_path, k_paths_full)
            print(f"  K-paths saved to: {k_paths_save_path}")
            print(f"  K-paths array shape: {k_paths_full.shape}")

        # Unconditional sampling for comparison
        if args.compare_unconditional:
            print("  Running unconditional comparison...")

            uncond_sampler = sampler.get_pc_sampler(
                graph=graph_module,
                noise=noise_module,
                batch_dims=(1, *args.data_dim),
                predictor=predictor,
                steps=args.path_length,
                eps=args.eps,
                device=device,
            )

            with torch.no_grad():
                uncond_samples, _ = uncond_sampler(model)

            uncond_samples_reshaped = uncond_samples.view(-1, 1, 28, 28).float() / 255.0
            uncond_grid = make_grid(
                uncond_samples_reshaped, nrow=1, padding=2, scale_each=True
            )
            save_image(
                uncond_grid,
                os.path.join(args.output_dir, f"uncond_run{run_idx}.png"),
            )

    # Save all collected PT samples as numpy array
    if len(all_pt_samples) > 0:
        all_samples_concatenated = np.concatenate(all_pt_samples, axis=0)
        total_samples_reshaped = all_samples_concatenated.reshape(-1, 1, 28, 28)
        samples_save_path = os.path.join(args.output_dir, "all_pt_samples.npy")
        np.save(samples_save_path, total_samples_reshaped)
        print(f"\nAll PT samples saved to: {samples_save_path}")
        print(f"Sample array shape: {total_samples_reshaped.shape}")
        print(f"Total PT runs: {args.num_runs}")
        print(f"Total samples collected: {len(total_samples_reshaped)}")

    # Save burn-in samples separately if any were collected
    if len(all_burn_in_samples) > 0:
        burn_in_samples_concatenated = np.concatenate(all_burn_in_samples, axis=0)
        total_burn_in_samples_reshaped = burn_in_samples_concatenated.reshape(
            -1, 1, 28, 28
        )
        burn_in_save_path = os.path.join(args.output_dir, "all_burn_in_samples.npy")
        np.save(burn_in_save_path, total_burn_in_samples_reshaped)
        print(f"Burn-in samples saved to: {burn_in_save_path}")
        print(f"Burn-in array shape: {total_burn_in_samples_reshaped.shape}")
        print(f"Total burn-in samples collected: {len(total_burn_in_samples_reshaped)}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
