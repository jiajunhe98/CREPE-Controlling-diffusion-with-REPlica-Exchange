"""
Unified script to compute perplexity for PT (Parallel Tempering) samples
using GPT2-XL with LoRA adapter weights.

Supports both toxicity and sentiment datasets via --dataset flag.
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_pt_json(json_path):
    """
    Process PT (Parallel Tempering) JSONL file and group sentences by configuration.
    Only collects samples where phase is "post_burn_in".

    Works for both toxicity and sentiment datasets (same JSONL format).

    Args:
        json_path: Path to the JSONL file

    Returns:
        List of dictionaries, each containing:
        - 'config': Configuration metadata (PT parameters)
        - 'sentences': List of post_burn_in sentences for this configuration
    """
    from collections import defaultdict

    # Group by configuration
    config_groups = defaultdict(list)

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)

            # Create configuration key based on PT parameters
            config_key = (
                data.get("cfg_temperature"),
                data.get("smc_temperature"),
                data.get("num_steps"),
                data.get("burn_in_steps"),
                data.get("path_length"),
                data.get("num_local_steps"),
                data.get("store_every_n_steps"),
                data.get("force_swap_at_one"),
                data.get("sampling_schedule"),
                data.get("eps"),
            )

            config_metadata = {
                "method": "pt",  # Parallel Tempering
                "cfg_temperature": data.get("cfg_temperature"),
                "smc_temperature": data.get("smc_temperature"),
                "num_steps": data.get("num_steps"),
                "burn_in_steps": data.get("burn_in_steps"),
                "path_length": data.get("path_length"),
                "num_local_steps": data.get("num_local_steps"),
                "store_every_n_steps": data.get("store_every_n_steps"),
                "force_swap_at_one": data.get("force_swap_at_one"),
                "sampling_schedule": data.get("sampling_schedule"),
                "eps": data.get("eps"),
            }

            # Extract post_burn_in sentences for this entry
            entry_sentences = []
            if "samples" in data:
                for sample in data["samples"]:
                    if sample.get("phase") == "post_burn_in" and "text" in sample:
                        entry_sentences.append(sample["text"])

            # Store configuration and sentences
            if (
                config_key not in config_groups
                or "config" not in config_groups[config_key]
            ):
                config_groups[config_key] = {
                    "config": config_metadata,
                    "sentences": [],
                }
            config_groups[config_key]["sentences"].extend(entry_sentences)

    # Convert to list format
    return list(config_groups.values())


def compute_perplexity(model, tokenizer, sentences, batch_size=8, device="cpu"):
    """
    Compute average perplexity over a list of sentences.

    Args:
        model: The language model
        tokenizer: The tokenizer
        sentences: List of sentences (strings)
        batch_size: Batch size for processing
        device: Device to run on

    Returns:
        Average perplexity across all sentences
    """
    model.eval()
    model.to(device)

    # Tokenize all sentences with padding and truncation
    encoded = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,  # GPT2-XL max sequence length
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Compute perplexity in batches
    num_samples = input_ids.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    total_perplexity = 0.0

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_input_ids = input_ids[start_idx:end_idx]
            batch_attention_mask = attention_mask[start_idx:end_idx]

            # Forward pass
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits

            # Transpose for cross_entropy: [batch, vocab, seq_len-1]
            logits = logits[..., :-1, :].transpose(-1, -2)
            targets = batch_input_ids[..., 1:]

            # Create mask for valid (non-padding) tokens
            # Shift attention mask to align with targets
            valid_mask = batch_attention_mask[..., 1:].bool()

            # Compute loss only on valid tokens
            losses = F.cross_entropy(logits, targets, reduction="none")
            # Mask out padding tokens
            masked_losses = losses * valid_mask.float()
            # Average over valid tokens only
            valid_token_count = valid_mask.sum(dim=-1)
            per_sample_loss = masked_losses.sum(dim=-1) / valid_token_count.clamp(min=1)
            # Convert to perplexity
            perplexity = per_sample_loss.exp().mean()

            total_perplexity += perplexity.item()

    # Average across batches
    avg_perplexity = total_perplexity / num_batches
    return avg_perplexity


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Compute perplexity for PT samples using GPT2-XL with LoRA weights"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["toxicity", "sentiment"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--path-to-json",
        type=str,
        required=True,
        help="Path to the JSONL file containing the PT samples",
    )
    parser.add_argument(
        "--lora-weights-path",
        type=str,
        required=True,
        help="Path to LoRA weights directory",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for perplexity computation",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to save the output JSON file with perplexity and metadata. If not provided, prepends 'ppl_' to input filename.",
    )
    args = parser.parse_args()

    # Set HuggingFace cache directory if provided
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    # Generate output filename if not provided
    if args.output is None:
        input_basename = os.path.basename(args.path_to_json)
        output_dir = os.path.dirname(args.path_to_json)
        output_basename = f"ppl_{input_basename.replace('.jsonl', '.json')}"
        args.output = (
            os.path.join(output_dir, output_basename) if output_dir else output_basename
        )

    # Get LoRA weights path
    lora_weights_path = args.lora_weights_path

    # Load base GPT2-XL model
    print("Loading GPT2-XL base model...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

    # GPT2 doesn't have a padding token by default, so we set it to EOS token
    # This is the standard approach for GPT2 when batch processing with padding
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter weights
    print(f"Loading LoRA adapter weights for {args.dataset}...")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    print("Model loaded successfully!")

    # Process PT JSONL file (same format for all datasets)
    print(f"Loading data from {args.path_to_json}...")
    config_groups = process_pt_json(args.path_to_json)

    print(f"Found {len(config_groups)} unique configuration(s)")

    # Compute perplexity for each configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_list = []

    for i, group in enumerate(config_groups, 1):
        config = group["config"]
        sentences = group["sentences"]

        print(f"\n[{i}/{len(config_groups)}] Processing configuration:")
        print(f"  Method: {config['method']}")
        print(
            f"  CFG temp: {config['cfg_temperature']}, SMC temp: {config['smc_temperature']}"
        )
        print(f"  Num steps: {config['num_steps']}, Burn-in: {config['burn_in_steps']}")
        print(
            f"  Path length: {config['path_length']}, Local steps: {config['num_local_steps']}"
        )
        print(f"  Sentences: {len(sentences)}")

        # Compute perplexity for this configuration
        perplexity = compute_perplexity(
            model, tokenizer, sentences, batch_size=args.batch_size, device=device
        )
        print(f"  Perplexity: {perplexity:.3f}")

        # Store results
        results_list.append(
            {
                "config": config,
                "perplexity": perplexity,
                "num_sentences": len(sentences),
            }
        )

    # Save all results to JSON
    output_data = {
        "results": results_list,
        "metadata": {
            "dataset": args.dataset,
            "input_jsonl_path": args.path_to_json,
            "lora_weights_path": lora_weights_path,
            "num_configurations": len(config_groups),
            "total_sentences": sum(len(g["sentences"]) for g in config_groups),
            "batch_size": args.batch_size,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {args.output}")
    print(f"Total configurations processed: {len(config_groups)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
