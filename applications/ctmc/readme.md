# Debiasing CFG with CREPE on CTMCs

Discrete diffusion experiments using CREPE on Continuous-Time Markov Chains (CTMCs).
Includes MNIST (image) and text (toxicity/sentiment control) experiments.

## 🛠️ Environment

### MNIST experiments
```bash
pip install torch torchvision jaxtyping matplotlib
```

### Text experiments (additionally require)
```bash
pip install transformers datasets peft einops flash-attn omegaconf jsonlines tqdm huggingface-hub
```

> **Note:** `flash-attn` requires CUDA and can be tricky to install. See [flash-attention](https://github.com/Dao-AILab/flash-attention) for instructions.


## 📚 Checkpoints

### MNIST
Train a model (Uniform graph, Geometric noise, CFG-enabled):
```bash
python mnist/train.py \
    --graph uniform --vocab-size 256 \
    --noise geometric --sigma-min 1e-3 --sigma-max 1.0 \
    --batch-size 256 --lr 4e-4 --num-steps 5000 \
    --ema-decay 0.99 --scaling-trick --cfg-train --cfg-dropout-prob 0.2 \
    --num-scales 3 --num-res-blocks 3 --ch-mult 1 2 4 --ch 64 \
    --save-dir ./checkpoints/mnist
```

### Text (base model)
The text experiments use the pretrained SEDD-medium model: [louaaron/sedd-medium](https://huggingface.co/louaaron/sedd-medium).

### Text (finetuned checkpoints)
Finetuned SEDD checkpoints for toxicity and sentiment control should be placed at a path passed via `--finetune-path`.

### Text (LoRA weights for evaluation)
LoRA-adapted GPT2-XL weights for perplexity evaluation should be placed at a path passed via `--lora-weights-path`.


## ✨ Reproducing Paper Results

All commands below use the exact hyperparameters from the paper. Replace placeholder paths (`<...>`) with your actual checkpoint locations.

### MNIST — SMC

Absorbing graph, Cosine noise, 64 particles, multinomial resampling:
```bash
python mnist/sample_smc.py \
    --model-path <path/to/mnist/checkpoint.pt> \
    --graph absorbing --vocab-size 256 \
    --noise cosine --noise-eps 1e-3 \
    --num-particles 64 --sampling-steps 100 \
    --proposal-strength 1.0 --smc-temperature 3.0 \
    --ess-threshold 0.5 --resample-fraction 0.2 \
    --resampling-method multinomial --sampling-threshold 0.0 \
    --num-runs 10 --num-iterations 8 \
    --compare-cfg --cfg-temperature 2.0 \
    --scaling-trick --denoise \
    --output-dir ./outputs/mnist_smc
```

### MNIST — Parallel Tempering (CREPE)

Absorbing graph, Cosine noise, 100-step paths, 1324 PT steps with 300 burn-in:
```bash
python mnist/sample_pt.py \
    --model-path <path/to/mnist/checkpoint.pt> \
    --graph absorbing --vocab-size 256 \
    --noise cosine --noise-eps 1e-3 \
    --path-length 100 --sampling-steps 100 --sampling-schedule linear \
    --num-steps 1324 --burn-in-steps 300 \
    --cfg-temperature 2.0 --smc-temperature 1.2 \
    --method prob --num-local-steps 0 \
    --keep-burn-in --num-runs 10 \
    --compare-cfg --compare-unconditional \
    --scaling-trick --denoise \
    --output-dir ./outputs/mnist_pt
```

### MNIST — RNE

Absorbing graph, Cosine noise, 64 particles, prob method:
```bash
python mnist/sample_rne.py \
    --model-path <path/to/mnist/checkpoint.pt> \
    --graph absorbing --vocab-size 256 \
    --noise cosine --noise-eps 1e-3 \
    --num-particles 64 --sampling-steps 100 \
    --proposal-strength 1.0 --smc-temperature 3.0 \
    --ess-threshold 0.5 --resample-fraction 0.2 \
    --resampling-method multinomial --sampling-threshold 0.0 \
    --method prob --denoise \
    --num-runs 10 --num-iterations 8 \
    --compare-cfg --cfg-temperature 2.0 \
    --scaling-trick \
    --output-dir ./outputs/mnist_rne
```

### Text — Toxicity (CFG + SMC)

1000 samples per class (toxic / nontoxic), partial resampling:
```bash
python text/generate_text.py \
    --task toxicity \
    --finetune-path <path/to/finetuned-sedd-toxicity> \
    --finetune-checkpoint-no 10 \
    --length 1024 --steps 100 --eps 1e-3 \
    --batch-size 50 --num-samples-per-class 1000 \
    --cfg-temperature 1.2 --smc-temperature 1.2 \
    --ess-threshold 0.3 --resample-fraction 0.8 \
    --resampling-method partial --sampling-threshold 0.0 \
    --output-file toxicity_smc_results.jsonl
```

To generate CFG-only samples (no SMC), add `--only-smc` to skip the CFG baseline, or run a separate CFG-only pass.

### Text — Toxicity (PT / CREPE)

256-step paths, 2300 PT steps, 300 burn-in:
```bash
python text/generate_text_pt.py \
    --task toxicity \
    --finetune-path <path/to/finetuned-sedd-toxicity> \
    --finetune-checkpoint-no 10 \
    --length 1024 --steps 100 --eps 1e-3 \
    --path-length 256 --num-steps 2300 --burn-in-steps 300 \
    --cfg-temperature 1.2 --smc-temperature 1.2 \
    --method prob --num-local-steps 0 \
    --keep-burn-in --batch-size-pt 16 \
    --store-k-paths 50 --store-every-n-steps 20
```

### Text — Sentiment (CFG + SMC)

1000 samples per class (positive / negative):
```bash
python text/generate_text.py \
    --task sentiment \
    --finetune-path <path/to/finetuned-sedd-sentiment> \
    --finetune-checkpoint-no 5 \
    --length 1024 --steps 100 --eps 1e-3 \
    --batch-size 50 --num-samples-per-class 1000 \
    --cfg-temperature 1.2 --smc-temperature 1.2 \
    --ess-threshold 0.3 --resample-fraction 0.8 \
    --resampling-method partial --sampling-threshold 0.0 \
    --output-file sentiment_smc_results.jsonl
```

### Text — Sentiment (PT / CREPE)
```bash
python text/generate_text_pt.py \
    --task sentiment \
    --finetune-path <path/to/finetuned-sedd-sentiment> \
    --finetune-checkpoint-no 5 \
    --length 1024 --steps 100 --eps 1e-3 \
    --path-length 256 --num-steps 2300 --burn-in-steps 300 \
    --cfg-temperature 1.2 --smc-temperature 1.2 \
    --method prob --num-local-steps 0 \
    --keep-burn-in --batch-size-pt 16 \
    --store-k-paths 50 --store-every-n-steps 20
```


## 📐 Evaluation

### Perplexity (PT samples)
```bash
python text/compute_perplexity_pt.py \
    --dataset toxicity \
    --path-to-json pt_results.jsonl \
    --lora-weights-path <path/to/lora-weights> \
    --batch_size 8 \
    --output perplexity_pt.json
```

### Perplexity (CFG / SMC samples)
```bash
python text/compute_perplexity_smc_cfg.py \
    --dataset toxicity \
    --path-to-json smc_results.jsonl \
    --lora-weights-path <path/to/lora-weights> \
    --batch_size 8 \
    --output perplexity_smc.json
```

### Sentiment accuracy
```bash
python text/compute_sentiment_accuracy.py \
    --path-to-json sentiment_results.jsonl \
    --output sentiment_accuracy.json
```
