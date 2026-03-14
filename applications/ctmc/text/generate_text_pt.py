#!/usr/bin/env python3
"""
Unified text generation using Parallel Tempering (PT).

Supports toxicity and sentiment tasks via --task flag.

Usage:
    python text/generate_text_pt.py --task toxicity --finetune-path /path/to/checkpoints
    python text/generate_text_pt.py --task sentiment --finetune-path /path/to/checkpoints
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for load_model, utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for lib imports

import jsonlines
import load_model
import numpy as np
import torch
from transformers import GPT2TokenizerFast

from lib import sampler
from lib.graph import Absorbing
from lib.noise import LogLinearNoise
from lib.sampler import EulerPredictor

# ---------------------------------------------------------------------------
# Task-specific configuration
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    "toxicity": {
        "classes": ["toxic", "nontoxic"],
        "prompts": {
            "toxic": "This text is toxic. ",
            "nontoxic": "This text is not toxic.",
        },
        "default_checkpoint_no": 10,
        "default_output": "toxicity_pt_results.jsonl",
        "default_save_paths_dir": "toxicity_pt_paths",
    },
    "sentiment": {
        "classes": ["positive", "negative"],
        "prompts": {
            "positive": "The sentiment of the text is positive",
            "negative": "The sentiment of the text is negative",
        },
        "default_checkpoint_no": 5,
        "default_output": "sentiment_pt_results.jsonl",
        "default_save_paths_dir": "sentiment_pt_paths",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_initial_path(
    graph,
    noise_module,
    predictor,
    uncond_model,
    cond_model,
    cond,
    time_steps,
    device,
    data_dim,
    cfg_temperature,
    eps,
):
    """Generate initial path using CFG proposal."""
    cfg_sampler = sampler.get_pc_sampler_cfg_finetune(
        graph=graph,
        noise=noise_module,
        batch_dims=(cond.shape[0], *data_dim),
        predictor=predictor,
        steps=len(time_steps) - 1,
        eps=eps,
        device=device,
        return_path=True,
    )

    final_samples, initial_samples, path_tensor = cfg_sampler(
        uncond_model, cond_model, cond, cfg_temperature=cfg_temperature
    )

    return path_tensor.squeeze(1)


def setup_save_dirs(base_dir, run_id):
    """Setup directory structure for saving paths and samples."""
    run_dir = Path(base_dir) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def generate_pt_filename(
    method,
    run_idx,
    class_label,
    cfg_temp=None,
    smc_temp=None,
    pt_steps=None,
    burn_in=None,
):
    """Generate descriptive filename for PT saved samples."""
    filename_parts = [method]

    if cfg_temp is not None:
        filename_parts.append(f"cfg{cfg_temp:.1f}")

    if smc_temp is not None:
        filename_parts.append(f"smc{smc_temp:.1f}")

    if pt_steps is not None:
        filename_parts.append(f"steps{pt_steps}")

    if burn_in is not None:
        filename_parts.append(f"burn{burn_in}")

    filename_parts.extend([f"run{run_idx}", f"class{class_label}"])

    return "_".join(filename_parts) + ".npy"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Parallel Tempering text generation"
    )
    parser.add_argument(
        "--task", required=True, choices=["toxicity", "sentiment"], help="Task to run"
    )
    parser.add_argument(
        "--base-model-path",
        default="louaaron/sedd-medium",
        help="HuggingFace model id or local path for base SEDD model",
    )
    parser.add_argument(
        "--finetune-path",
        required=True,
        help="Path to finetuned SEDD checkpoint directory",
    )
    parser.add_argument(
        "--finetune-checkpoint-no",
        type=int,
        default=None,
        help="Checkpoint number (default: task-specific)",
    )
    parser.add_argument("--length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--eps", type=float, default=1e-3, help="Noise epsilon")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the models"
    )
    parser.add_argument("--hf-cache-dir", default=None, help="HuggingFace cache dir")

    # PT-specific parameters
    parser.add_argument(
        "--path-length", type=int, default=256, help="Number of time steps in the path"
    )
    parser.add_argument(
        "--num-steps", type=int, default=2300, help="Number of PT iterations"
    )
    parser.add_argument(
        "--burn-in-steps", type=int, default=300, help="Burn-in steps to discard"
    )
    parser.add_argument(
        "--cfg-temperature", type=float, default=1.2, help="CFG temperature"
    )
    parser.add_argument(
        "--smc-temperature", type=float, default=1.2, help="SMC/RNE temperature"
    )
    parser.add_argument(
        "--method",
        default="prob",
        choices=["prob", "score"],
        help="log_R computation method",
    )
    parser.add_argument(
        "--num-local-steps",
        type=int,
        default=0,
        help="Local CTMC moves after accepted Metropolis updates",
    )
    parser.add_argument(
        "--force-swap-at-one",
        action="store_true",
        default=False,
        help="Force all swaps to be accepted (debug)",
    )
    parser.add_argument(
        "--keep-burn-in",
        action="store_true",
        default=True,
        help="Keep samples during burn-in period",
    )
    parser.add_argument(
        "--batch-size-pt",
        type=int,
        default=16,
        help="Batch size for PT pair processing",
    )
    parser.add_argument(
        "--store-k-paths",
        type=int,
        default=50,
        help="Number of intermediate paths to store",
    )
    parser.add_argument(
        "--store-every-n-steps",
        type=int,
        default=20,
        help="Store paths every N steps",
    )

    parser.add_argument(
        "--output-file", default=None, help="Output JSONL path (default: task-specific)"
    )
    parser.add_argument(
        "--save-paths-dir",
        default=None,
        help="Directory for saving .npy paths (default: task-specific)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    task_cfg = TASK_CONFIGS[args.task]

    classes = task_cfg["classes"]
    prompts = task_cfg["prompts"]
    checkpoint_no = args.finetune_checkpoint_no or task_cfg["default_checkpoint_no"]
    output_file_default = args.output_file or task_cfg["default_output"]
    save_paths_dir = args.save_paths_dir or task_cfg["default_save_paths_dir"]

    num_runs = len(classes)

    print(f"=== Text Generation with Parallel Tempering ({args.task}) ===")
    print(f"Path length: {args.path_length}")
    print(f"PT steps: {args.num_steps}, Burn-in: {args.burn_in_steps}")
    print(f"Running PT on {num_runs} prompts (classes)")
    expected_samples = (
        args.num_steps - args.burn_in_steps if not args.keep_burn_in else args.num_steps
    )
    print(f"Expected samples per prompt: ~{expected_samples}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup save directories
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = setup_save_dirs(save_paths_dir, run_id)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.hf_cache_dir)

    # Instantiate components directly
    graph_module = Absorbing(dim=50257)
    noise_module = LogLinearNoise(eps=args.eps)
    predictor = EulerPredictor(graph_module, noise_module)

    # Load models
    uncond_model = load_model.load_model(args.base_model_path, device)
    cond_model = load_model.load_finetune_model(
        args.finetune_path, checkpoint_no, device
    )

    if args.compile:
        uncond_model = torch.compile(uncond_model)
        cond_model = torch.compile(cond_model)

    # Only linear schedule is supported for PT
    sampling_schedule = "linear"

    # Create time discretization for paths
    time_steps = sampler.sampling_schedule_grid(
        sampling_schedule, args.path_length, args.eps, device
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
        store_k_paths=args.store_k_paths,
        store_every_n_steps=args.store_every_n_steps,
        num_local_steps=args.num_local_steps,
        force_swap_at_one=args.force_swap_at_one,
        method=args.method,
        keep_burn_in=args.keep_burn_in,
        score_fn_type="finetune",
        batch_size_pt=args.batch_size_pt,
    )

    # Build output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = random.randint(1000, 9999)
    output_path = Path(output_file_default)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filename_parts = [
        output_path.stem,
        timestamp,
        f"cfg{args.cfg_temperature:.1f}".replace(".", "p"),
        f"smc{args.smc_temperature:.1f}".replace(".", "p"),
        f"pl{args.path_length}",
        f"steps{args.num_steps}",
        str(random_id),
    ]
    output_file = output_path.parent / f"{'_'.join(filename_parts)}{output_path.suffix}"

    # Collect all PT samples for analysis
    all_pt_samples = []
    all_burn_in_samples = []

    data_dim = [args.length]

    print(f"Writing results to: {output_file}")

    with jsonlines.open(output_file, mode="w") as writer:
        for run_idx in range(num_runs):
            class_label = classes[run_idx % len(classes)]
            prompt = prompts[class_label]

            print(f"\nPT Run {run_idx + 1}/{num_runs} - Prompt: {class_label}")

            run_start_time = time.time()

            # Create conditioning (tokenize prompt)
            tokenized_prompt = tokenizer.encode(prompt)
            cond = torch.tensor([tokenized_prompt], device=device)

            # Generate initial path
            print("  Generating initial path...")
            initial_path = generate_initial_path(
                graph_module,
                noise_module,
                predictor,
                uncond_model,
                cond_model,
                cond,
                time_steps,
                device,
                data_dim,
                args.cfg_temperature,
                args.eps,
            )

            print(f"  Initial path shape: {initial_path.shape}")
            print(f"  time_steps shape: {time_steps.shape}")
            assert initial_path.shape[0] == time_steps.shape[0], (
                f"initial_path: {initial_path.shape}, time_steps: {time_steps.shape}"
            )

            # Log initial path final sample
            initial_sample = initial_path[-1].cpu()
            initial_text = tokenizer.decode(initial_sample.tolist())
            print(f"  Initial sample: {initial_text[:120]}...")

            # Run PT sampling
            print("  Running parallel tempering...")
            with torch.no_grad():
                result = pt_sampler(
                    uncond_model,
                    cond_model,
                    cond,
                    initial_path,
                    traces=[
                        "accept_mask",
                        "log_weights_reverse",
                        "log_weights_forward",
                    ],
                )

            # Log acceptance statistics
            mean_acceptance_rate = None
            if result.accept_count_per_timestep is not None:
                accept_rates_per_timestep = (
                    result.accept_count_per_timestep.float() / args.num_steps
                )
                mean_acceptance_rate = accept_rates_per_timestep.mean().item()
                print(f"  Mean acceptance rate: {mean_acceptance_rate:.3f}")

            # Log weight statistics
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
                mean_weights_product = torch.exp(mean_log_weights_reverse) * torch.exp(
                    mean_log_weights_forward
                )
                print(
                    f"  Mean weight product: "
                    f"min={mean_weights_product.min().item():.4f}, "
                    f"max={mean_weights_product.max().item():.4f}, "
                    f"mean={mean_weights_product.mean().item():.4f}"
                )

            # Combine burn-in and post-burn-in samples with step tracking
            all_samples_with_steps = []

            # Process burn-in samples if available
            if result.burn_in_samples is not None and len(result.burn_in_samples) > 0:
                print(
                    f"  Collected {len(result.burn_in_samples)} samples during burn-in"
                )
                burn_in_samples_tensor = torch.stack(result.burn_in_samples)
                burn_in_samples_np = burn_in_samples_tensor.cpu().numpy()
                all_burn_in_samples.append(burn_in_samples_np)

                for i, sample in enumerate(burn_in_samples_tensor):
                    step = i
                    text = tokenizer.decode(sample.cpu().tolist())
                    all_samples_with_steps.append(
                        {"step": step, "text": text, "phase": "burn_in"}
                    )

                    if i < 3:
                        print(f"    Burn-in sample {i}: {text[:120]}...")

            # Process post-burn-in samples
            print(f"  Collected {len(result.samples)} samples after burn-in")
            if len(result.samples) > 0:
                samples_tensor = torch.stack(result.samples)
                samples_np = samples_tensor.cpu().numpy()
                all_pt_samples.append(samples_np)

                for i, sample in enumerate(samples_tensor):
                    step = (
                        len(result.burn_in_samples) + i if result.burn_in_samples else i
                    )
                    text = tokenizer.decode(sample.cpu().tolist())
                    all_samples_with_steps.append(
                        {"step": step, "text": text, "phase": "post_burn_in"}
                    )

                    if i < 3:
                        print(f"    Post-burn-in sample {i}: {text[:120]}...")

            run_elapsed_time = time.time() - run_start_time

            # Write to JSONL
            if len(all_samples_with_steps) > 0:
                writer.write(
                    {
                        "method": "pt",
                        "run_idx": run_idx,
                        "class": class_label,
                        "prompt": prompt,
                        "cfg_temperature": float(args.cfg_temperature),
                        "smc_temperature": float(args.smc_temperature),
                        "num_steps": int(args.num_steps),
                        "burn_in_steps": int(args.burn_in_steps),
                        "path_length": int(args.path_length),
                        "num_local_steps": int(args.num_local_steps),
                        "store_every_n_steps": int(args.store_every_n_steps),
                        "force_swap_at_one": bool(args.force_swap_at_one),
                        "sampling_schedule": sampling_schedule,
                        "eps": float(args.eps),
                        "num_samples": len(all_samples_with_steps),
                        "num_burn_in_samples": len(result.burn_in_samples)
                        if result.burn_in_samples
                        else 0,
                        "num_post_burn_in_samples": len(result.samples),
                        "mean_acceptance_rate": mean_acceptance_rate,
                        "elapsed_time_seconds": float(run_elapsed_time),
                        "timestamp": datetime.now().isoformat(),
                        "samples": all_samples_with_steps,
                    }
                )

            # Save final path as numpy array
            final_path_np = result.final_path.cpu().numpy()
            filename = generate_pt_filename(
                method="final_path",
                run_idx=run_idx,
                class_label=class_label,
                cfg_temp=args.cfg_temperature,
                smc_temp=args.smc_temperature,
                pt_steps=args.num_steps,
                burn_in=args.burn_in_steps,
            )
            save_path = save_dir / filename
            np.save(save_path, final_path_np)

            # Save stored k paths if available
            if result.k_paths_storage is not None:
                print(
                    f"  Saving {result.k_paths_storage.shape[0]} stored intermediate paths..."
                )
                k_paths_np = result.k_paths_storage.cpu().numpy()

                k_paths_filename = generate_pt_filename(
                    method="k_paths",
                    run_idx=run_idx,
                    class_label=class_label,
                    cfg_temp=args.cfg_temperature,
                    smc_temp=args.smc_temperature,
                    pt_steps=args.num_steps,
                    burn_in=args.burn_in_steps,
                )

                k_paths_save_path = save_dir / k_paths_filename
                np.save(k_paths_save_path, k_paths_np)
                print(f"  K-paths saved to: {k_paths_save_path}")
                print(f"  K-paths array shape: {k_paths_np.shape}")

    # Save all collected PT samples as numpy array
    if len(all_pt_samples) > 0:
        all_samples_concatenated = np.concatenate(all_pt_samples, axis=0)
        samples_save_path = save_dir / "all_pt_samples.npy"
        np.save(samples_save_path, all_samples_concatenated)
        print(f"\nAll PT samples saved to: {samples_save_path}")
        print(f"Sample array shape: {all_samples_concatenated.shape}")
        print(f"Total PT runs: {num_runs}")
        print(f"Total samples collected: {len(all_samples_concatenated)}")

    # Save burn-in samples separately if any were collected
    if len(all_burn_in_samples) > 0:
        burn_in_samples_concatenated = np.concatenate(all_burn_in_samples, axis=0)
        burn_in_save_path = save_dir / "all_burn_in_samples.npy"
        np.save(burn_in_save_path, burn_in_samples_concatenated)
        print(f"Burn-in samples saved to: {burn_in_save_path}")
        print(f"Burn-in array shape: {burn_in_samples_concatenated.shape}")
        print(f"Total burn-in samples collected: {len(burn_in_samples_concatenated)}")

    print(f"\nResults saved to: {output_file}")
    print(f"Paths saved to: {save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
