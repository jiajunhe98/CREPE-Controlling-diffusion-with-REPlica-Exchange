#!/usr/bin/env python3
"""
Unified text generation using CFG and CFG-SMC sampling.

Supports toxicity and sentiment tasks via --task flag.

Usage:
    python text/generate_text.py --task toxicity --finetune-path /path/to/checkpoints
    python text/generate_text.py --task sentiment --finetune-path /path/to/checkpoints --only-smc
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
        "default_output": "toxicity_smc_results.jsonl",
    },
    "sentiment": {
        "classes": ["positive", "negative"],
        "prompts": {
            "positive": "The sentiment of the text is positive",
            "negative": "The sentiment of the text is negative",
        },
        "default_checkpoint_no": 5,
        "default_output": "sentiment_smc_results.jsonl",
    },
}


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def run_cfg_smc_sample(
    cond,
    uncond_model,
    cond_model,
    graph,
    noise_module,
    predictor,
    strength,
    twist_strength,
    device,
    *,
    length,
    sampling_steps,
    eps,
    ess_threshold,
    resample_fraction,
    resampling_method,
    sampling_threshold,
):
    sampling_fn = sampler.get_cfg_smc_sampler(
        graph=graph,
        noise=noise_module,
        batch_dims=(cond.shape[0], length),
        predictor=predictor,
        steps=sampling_steps,
        proposal_strength=strength,
        smc_temperature=twist_strength,
        eps=eps,
        ess_threshold=ess_threshold,
        resample_fraction=resample_fraction,
        resampling_method=resampling_method,
        sampling_threshold=sampling_threshold,
        score_fn_type="finetune",
        device=device,
    )
    result = sampling_fn(uncond_model, cond_model, cond)
    return result.particles


def run_cfg_sample(
    cond,
    uncond_model,
    cond_model,
    graph,
    noise_module,
    predictor,
    strength,
    device,
    *,
    length,
    steps,
    eps,
):
    sampling_fn = sampler.get_pc_sampler_cfg_finetune(
        graph=graph,
        noise=noise_module,
        batch_dims=(cond.shape[0], length),
        predictor=predictor,
        steps=steps,
        eps=eps,
        device=device,
    )
    samples, _ = sampling_fn(uncond_model, cond_model, cond, cfg_temperature=strength)
    return samples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CFG + CFG-SMC text generation"
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
    parser.add_argument(
        "--steps", type=int, default=100, help="Sampling steps for both CFG and SMC"
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=1000,
        help="Total samples per class",
    )
    parser.add_argument(
        "--cfg-temperature", type=float, default=1.2, help="CFG temperature"
    )
    parser.add_argument(
        "--smc-temperature", type=float, default=1.2, help="SMC temperature"
    )
    parser.add_argument(
        "--ess-threshold", type=float, default=0.3, help="ESS resampling threshold"
    )
    parser.add_argument(
        "--resample-fraction", type=float, default=0.8, help="Fraction to resample"
    )
    parser.add_argument(
        "--resampling-method",
        default="partial",
        help="Resampling method: partial or multinomial",
    )
    parser.add_argument(
        "--sampling-threshold",
        type=float,
        default=0.0,
        help="Min fraction of steps before resampling",
    )
    parser.add_argument("--eps", type=float, default=1e-3, help="Noise epsilon")
    parser.add_argument(
        "--only-smc", action="store_true", help="Skip standard CFG sampling"
    )
    parser.add_argument(
        "--output-file", default=None, help="Output JSONL path (default: task-specific)"
    )
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the models"
    )
    parser.add_argument("--hf-cache-dir", default=None, help="HuggingFace cache dir")
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

    num_runs = len(classes)
    num_iterations = args.num_samples_per_class // args.batch_size

    print(f"=== Text Generation with CFG and CFG-SMC ({args.task}) ===")
    print(f"Num runs: {num_runs}")
    print(f"Samples per class: {args.num_samples_per_class}")
    print(f"Batch size: {args.batch_size}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.hf_cache_dir)

    # Instantiate components directly
    graph_module = Absorbing(dim=50257)
    noise_module = LogLinearNoise(eps=args.eps)
    predictor = EulerPredictor(graph_module, noise_module)

    # Load models
    model = load_model.load_model(args.base_model_path, device)
    finetune_model = load_model.load_finetune_model(
        args.finetune_path, checkpoint_no, device
    )

    if args.compile:
        model = torch.compile(model)
        finetune_model = torch.compile(finetune_model)

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
        f"samples{args.num_samples_per_class}",
        f"batch{args.batch_size}",
        str(random_id),
    ]
    output_file = output_path.parent / f"{'_'.join(filename_parts)}{output_path.suffix}"

    print(f"Writing results to: {output_file}")
    print(f"Num iterations per run: {num_iterations}")

    with jsonlines.open(output_file, mode="w") as writer:
        for run_idx in range(num_runs):
            class_label = classes[run_idx % len(classes)]
            prompt = prompts[class_label]

            print(f"\nRun {run_idx + 1}/{num_runs} - Prompt: {class_label}")

            run_start_time = time.time()

            tokenized_prompt = tokenizer.encode(prompt)
            cond = torch.tensor([tokenized_prompt] * args.batch_size, device=device)

            # ===== CFG-SMC Sampling =====
            print(
                f"  Running CFG-SMC with cfg_temperature={args.cfg_temperature}, "
                f"smc_temperature={args.smc_temperature}"
            )
            all_samples_with_steps = []
            global_step = 0

            for iteration in range(num_iterations):
                if iteration % 10 == 0:
                    print(f"    CFG-SMC iteration {iteration + 1}/{num_iterations}")

                cfg_smc_samples = run_cfg_smc_sample(
                    cond,
                    model,
                    finetune_model,
                    graph_module,
                    noise_module,
                    predictor,
                    args.cfg_temperature,
                    args.smc_temperature,
                    device,
                    length=args.length,
                    sampling_steps=args.steps,
                    eps=args.eps,
                    ess_threshold=args.ess_threshold,
                    resample_fraction=args.resample_fraction,
                    resampling_method=args.resampling_method,
                    sampling_threshold=args.sampling_threshold,
                )

                for sample in cfg_smc_samples:
                    text = tokenizer.decode(sample.tolist())
                    all_samples_with_steps.append(
                        {"step": global_step, "text": text, "phase": ""}
                    )
                    global_step += 1

            run_elapsed_time = time.time() - run_start_time

            # Write CFG-SMC results
            writer.write(
                {
                    "method": "cfg_smc",
                    "run_idx": run_idx,
                    "class": class_label,
                    "prompt": prompt,
                    "cfg_temperature": float(args.cfg_temperature),
                    "smc_temperature": float(args.smc_temperature),
                    "steps": int(args.steps),
                    "length": int(args.length),
                    "batch_size": int(args.batch_size),
                    "num_iterations": int(num_iterations),
                    "ess_threshold": float(args.ess_threshold),
                    "resample_fraction": float(args.resample_fraction),
                    "resampling_method": str(args.resampling_method),
                    "sampling_threshold": float(args.sampling_threshold),
                    "eps": float(args.eps),
                    "num_samples": len(all_samples_with_steps),
                    "elapsed_time_seconds": float(run_elapsed_time),
                    "timestamp": datetime.now().isoformat(),
                    "samples": all_samples_with_steps,
                }
            )

            # Print a few samples
            print(
                f"  CFG-SMC completed: {len(all_samples_with_steps)} samples "
                f"in {run_elapsed_time:.2f}s"
            )
            for i in range(min(3, len(all_samples_with_steps))):
                preview = all_samples_with_steps[i]["text"][:120]
                print(f"    Sample {i}: {preview}...")

            # ===== Standard CFG Sampling =====
            if not args.only_smc:
                print(
                    f"  Running standard CFG with cfg_temperature={args.cfg_temperature}"
                )
                all_samples_with_steps = []
                global_step = 0
                run_start_time = time.time()

                for iteration in range(num_iterations):
                    if iteration % 10 == 0:
                        print(f"    CFG iteration {iteration + 1}/{num_iterations}")

                    cfg_samples = run_cfg_sample(
                        cond,
                        model,
                        finetune_model,
                        graph_module,
                        noise_module,
                        predictor,
                        args.cfg_temperature,
                        device,
                        length=args.length,
                        steps=args.steps,
                        eps=args.eps,
                    )

                    for sample in cfg_samples:
                        text = tokenizer.decode(sample.tolist())
                        all_samples_with_steps.append(
                            {"step": global_step, "text": text, "phase": ""}
                        )
                        global_step += 1

                run_elapsed_time = time.time() - run_start_time

                writer.write(
                    {
                        "method": "cfg",
                        "run_idx": run_idx,
                        "class": class_label,
                        "prompt": prompt,
                        "cfg_temperature": float(args.cfg_temperature),
                        "steps": int(args.steps),
                        "length": int(args.length),
                        "eps": float(args.eps),
                        "batch_size": int(args.batch_size),
                        "num_iterations": int(num_iterations),
                        "num_samples": len(all_samples_with_steps),
                        "elapsed_time_seconds": float(run_elapsed_time),
                        "timestamp": datetime.now().isoformat(),
                        "samples": all_samples_with_steps,
                    }
                )

                print(
                    f"  CFG completed: {len(all_samples_with_steps)} samples "
                    f"in {run_elapsed_time:.2f}s"
                )
                for i in range(min(3, len(all_samples_with_steps))):
                    preview = all_samples_with_steps[i]["text"][:120]
                    print(f"    Sample {i}: {preview}...")

    print(f"\nResults saved to: {output_file}")
    print("Done.")


if __name__ == "__main__":
    main()
