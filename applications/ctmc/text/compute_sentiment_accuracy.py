"""
Script to compute sentiment classification accuracy for generated samples.
Evaluates whether generated samples match their target sentiment (positive/negative)
using a finetuned DistilBERT classifier.

Supports PT, SMC, and CFG sampling methods.
"""

import argparse
import json
import os
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSIFIER_MODEL = "Kai1014/distilbert-finetuned"

# Label mapping: classifier outputs logits for [negative, positive]
LABEL_TO_IDX = {"negative": 0, "positive": 1}
IDX_TO_LABEL = {0: "negative", 1: "positive"}


def process_pt_samples(json_path):
    """
    Process PT (Parallel Tempering) JSONL file.
    Only collects samples where phase is "post_burn_in".

    Args:
        json_path: Path to the JSONL file

    Returns:
        List of dicts with "text" and "target" keys
    """
    samples = []

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            target_class = data.get("class")

            if target_class not in LABEL_TO_IDX:
                continue

            if "samples" in data:
                for sample in data["samples"]:
                    if sample.get("phase") == "post_burn_in" and "text" in sample:
                        samples.append({"text": sample["text"], "target": target_class})

    return samples


def process_smc_cfg_samples(json_path):
    """
    Process SMC/CFG JSONL file.
    Uses all samples (no phase filtering).

    Args:
        json_path: Path to the JSONL file

    Returns:
        List of dicts with "text" and "target" keys
    """
    samples = []

    with open(json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            target_class = data.get("class")

            if target_class not in LABEL_TO_IDX:
                continue

            if "samples" in data:
                for sample in data["samples"]:
                    if "text" in sample:
                        samples.append({"text": sample["text"], "target": target_class})

    return samples


def classify_texts(classifier, tokenizer, texts, batch_size, device):
    """
    Classify texts using the sentiment classifier.

    Args:
        classifier: The classification model
        tokenizer: The tokenizer
        texts: List of text strings
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        List of predicted labels (0=negative, 1=positive)
    """
    classifier.eval()
    predictions = []

    num_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Classifying"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Tokenize batch
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predictions (argmax)
            batch_preds = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(batch_preds)

    return predictions


def compute_metrics(predictions, targets):
    """
    Compute accuracy metrics and confusion matrix.

    Args:
        predictions: List of predicted labels (0 or 1)
        targets: List of target labels (0 or 1)

    Returns:
        Dict with accuracy metrics and confusion matrix
    """
    # Convert targets to indices
    target_indices = [LABEL_TO_IDX[t] for t in targets]

    # Count confusion matrix components
    tp = tn = fp = fn = 0
    for pred, target in zip(predictions, target_indices):
        if pred == 1 and target == 1:
            tp += 1
        elif pred == 0 and target == 0:
            tn += 1
        elif pred == 1 and target == 0:
            fp += 1
        else:  # pred == 0 and target == 1
            fn += 1

    total = len(predictions)
    overall_accuracy = (tp + tn) / total if total > 0 else 0.0

    # Per-class accuracy
    positive_total = tp + fn
    negative_total = tn + fp
    positive_accuracy = tp / positive_total if positive_total > 0 else 0.0
    negative_accuracy = tn / negative_total if negative_total > 0 else 0.0

    return {
        "accuracy": overall_accuracy,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "total_samples": total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute sentiment classification accuracy for generated samples"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["pt", "smc", "cfg"],
        help="Sampling method: 'pt' (Parallel Tempering), 'smc', or 'cfg'",
    )
    parser.add_argument(
        "--path-to-json",
        type=str,
        required=True,
        help="Path to the JSONL file containing generated samples",
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
        default=32,
        help="Batch size for classification inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Output JSON path. Defaults to 'acc_{input}.json'",
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
        output_basename = f"acc_{input_basename.replace('.jsonl', '.json')}"
        args.output = (
            os.path.join(output_dir, output_basename) if output_dir else output_basename
        )

    # Load classifier
    print(f"Loading classifier: {CLASSIFIER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(device)
    print(f"Classifier loaded on {device}")

    # Process samples based on method
    print(f"Loading samples from {args.path_to_json}")
    if args.method == "pt":
        samples = process_pt_samples(args.path_to_json)
    else:  # smc or cfg
        samples = process_smc_cfg_samples(args.path_to_json)

    print(f"Loaded {len(samples)} samples")

    if len(samples) == 0:
        print("No samples found. Check JSONL format and 'class' field.")
        return

    # Extract texts and targets
    texts = [s["text"] for s in samples]
    targets = [s["target"] for s in samples]

    # Classify texts
    print("Running classification...")
    predictions = classify_texts(classifier, tokenizer, texts, args.batch_size, device)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    # Add metadata
    output_data = {
        **metrics,
        "metadata": {
            "method": args.method,
            "input_jsonl_path": args.path_to_json,
            "classifier_model": CLASSIFIER_MODEL,
            "batch_size": args.batch_size,
            "device": device,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Results saved to {args.output}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Positive Accuracy: {metrics['positive_accuracy']:.4f}")
    print(f"Negative Accuracy: {metrics['negative_accuracy']:.4f}")
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
