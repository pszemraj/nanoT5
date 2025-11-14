# Usage Guide

This document describes how to run pre-training, fine-tuning, and evaluation with nanoT5.

## Quick Start

### Minimal Pre-Training Example

```bash
# Pre-train with default config (65,536 steps)
python -m nanoT5.main
```

This will:
1. Load T5-base config from HuggingFace
2. Initialize model with random weights
3. Stream C4 dataset
4. Train with AdamWScale optimizer and cosine LR schedule
5. Save checkpoint at the end in `outputs/YYYY-MM-DD/HH-MM-SS/`

### Using Pre-Built Configs

```bash
# 16-hour pre-training (recommended)
python -m nanoT5.main task=pt_16h

# Other time budgets
python -m nanoT5.main task=pt_4h   # 4 hours
python -m nanoT5.main task=pt_8h   # 8 hours
python -m nanoT5.main task=pt_12h  # 12 hours
python -m nanoT5.main task=pt_20h  # 20 hours
python -m nanoT5.main task=pt_24h  # 24 hours
```

Each task config sets `optim.total_steps`, `optim.warmup_steps`, `optim.batch_size`, and `optim.grad_acc` appropriately.

## Pre-Training Workflows

### Standard Pre-Training

Pre-training uses streaming C4 dataset with span-corruption objective:

```bash
python -m nanoT5.main \
    mode=pt \
    model.klass=local_t5 \
    model.name=google/t5-v1_1-base \
    model.random_init=true \
    optim.name=adamwscale \
    optim.lr_scheduler=cosine \
    optim.base_lr=2e-2 \
    optim.batch_size=128 \
    optim.total_steps=65536 \
    optim.warmup_steps=10000 \
    precision=bf16 \
    device=gpu
```

**Required config**:
- `mode: pt` - Pre-training mode
- `model.random_init: true` - Initialize from scratch
- `model.klass: local_t5` - Use custom T5 implementation

**Key hyperparameters**:
- `optim.batch_size` - Functional batch size (total across accumulation steps)
- `optim.grad_acc` - Gradient accumulation steps (micro_batch_size = batch_size / grad_acc)
- `optim.total_steps` - Total optimization steps
- `data.input_length: 512` - Sequence length after masking
- `data.mlm_probability: 0.15` - Fraction of tokens to mask
- `data.mean_noise_span_length: 3.0` - Average span length for masking

### Pre-Training with Different Optimizers

```bash
# AdamWScale (recommended, default)
python -m nanoT5.main optim.name=adamwscale

# Standard AdamW (not recommended, worse convergence)
python -m nanoT5.main optim.name=adamw

# Adafactor (T5 original optimizer)
python -m nanoT5.main optim.name=adafactor
```

### Pre-Training with Different LR Schedules

```bash
# Cosine schedule (recommended, default)
python -m nanoT5.main optim.lr_scheduler=cosine optim.final_cosine=1e-5

# Inverse-square-root (T5 original)
python -m nanoT5.main optim.lr_scheduler=legacy

# Constant LR (fine-tuning default)
python -m nanoT5.main optim.lr_scheduler=constant
```

### Pre-Training on CPU

```bash
python -m nanoT5.main device=cpu precision=no
```

**Warning**: CPU training is extremely slow (~74 hours for full pre-training vs. 10 hours on A100).

### Adjusting for GPU Memory

If you run out of memory, increase gradient accumulation:

```bash
# For 24GB GPU (increase grad_acc from 1 to 2)
python -m nanoT5.main task=pt_16h optim.grad_acc=2

# Ensure batch_size is divisible by grad_acc
python -m nanoT5.main optim.batch_size=128 optim.grad_acc=4  # micro_batch = 32
```

### Disabling PyTorch Compile

If `torch.compile` causes issues:

```bash
python -m nanoT5.main model.compile=false
```

## Fine-Tuning Workflows

Fine-tuning uses Super-Natural Instructions (SNI) dataset. First, clone the SNI dataset:

```bash
git clone https://github.com/allenai/natural-instructions.git data
```

### Fine-Tuning from HuggingFace Checkpoint

```bash
# Fine-tune google/t5-v1_1-base from HuggingFace
python -m nanoT5.main task=ft \
    model.klass=hf_t5 \
    model.name=google/t5-v1_1-base \
    model.random_init=false \
    model.checkpoint_path=""
```

### Fine-Tuning from Your Pre-Trained Checkpoint

```bash
# Fine-tune your own checkpoint (from pre-training)
python -m nanoT5.main task=ft \
    model.klass=local_t5 \
    model.random_init=false \
    model.checkpoint_path=/path/to/outputs/YYYY-MM-DD/HH-MM-SS/checkpoint-pt-65536/pytorch_model.bin
```

**Important**: Use `model.klass=local_t5` for checkpoints trained with `MyT5`, and `model.klass=hf_t5` for HuggingFace checkpoints.

### Fine-Tuning from Random Initialization (Baseline)

```bash
# Random init baseline (for comparison)
python -m nanoT5.main task=ft \
    model.random_init=true \
    model.checkpoint_path=""
```

### Fine-Tuning Config Details

The `task=ft` config (`nanoT5/configs/task/ft.yaml`) sets:

```yaml
mode: 'ft'
precision: 'no'  # No mixed precision for fine-tuning
model:
  klass: hf_t5
data:
  max_seq_len: 1024
  max_target_len: 128
  max_num_instances_per_task: 100
  add_task_definition: True
  num_pos_examples: 2
optim:
  name: adamw
  base_lr: 5e-5
  batch_size: 8
  epochs: 2
  lr_scheduler: constant
  grad_acc: 1
eval:
  steps: 200
```

You can override these:

```bash
python -m nanoT5.main task=ft \
    optim.epochs=5 \
    optim.base_lr=1e-4 \
    data.num_pos_examples=3
```

## Evaluation Workflows

### Evaluation During Training

Evaluation happens automatically during training at `eval.every_steps` intervals:

```bash
# Evaluate every 1000 steps
python -m nanoT5.main eval.every_steps=1000 eval.steps=500
```

For pre-training, default is `eval.every_steps=100000` (once at the end). For fine-tuning, it depends on dataset size.

### Eval-Only Mode

To evaluate a checkpoint without training:

```bash
python -m nanoT5.main \
    eval_only=true \
    model.checkpoint_path=/path/to/pytorch_model.bin \
    eval.steps=500
```

**Note**: `eval_only` runs loss and accuracy computation. For generation metrics (ROUGE-L), use `predict_only` (fine-tuning only).

### Predict-Only Mode (Fine-Tuning)

Generate predictions and compute ROUGE-L:

```bash
python -m nanoT5.main \
    task=ft \
    predict_only=true \
    model.checkpoint_path=/path/to/pytorch_model.bin
```

This runs the generation loop (`predict()` function in `train_utils.py:126-172`) and computes ROUGE-L on the test set.

## Configuration System

### Config Hierarchy

Configs are composed in this order:
1. `default.yaml` - Base config
2. `task/*.yaml` - Task-specific overrides (pt, pt_16h, ft, etc.)
3. `local_env/*.yaml` - Environment-specific overrides
4. CLI arguments - Highest priority

### Common Config Overrides

```bash
# Change optimizer learning rate
python -m nanoT5.main optim.base_lr=1e-3

# Change model size (requires different model.name)
python -m nanoT5.main model.name=google/t5-v1_1-large

# Change batch size and grad accumulation
python -m nanoT5.main optim.batch_size=256 optim.grad_acc=2

# Change total steps
python -m nanoT5.main optim.total_steps=10000

# Change precision
python -m nanoT5.main precision=tf32  # or bf16, no

# Change random seed
python -m nanoT5.main seed=42

# Disable WandB
python -m nanoT5.main logging.use_wandb=false
```

### Multi-Field Override Example

```bash
python -m nanoT5.main \
    task=pt_16h \
    optim.name=adamwscale \
    optim.lr_scheduler=cosine \
    optim.base_lr=2e-2 \
    optim.final_cosine=1e-5 \
    precision=bf16 \
    model.compile=true \
    logging.use_wandb=true \
    logging.wandb_config.project=my_project \
    logging.wandb_config.entity=my_username
```

### Accessing Config Values

The config is available as `args` (OmegaConf DictConfig) throughout the code:

```python
# Access nested values
lr = args.optim.base_lr
model_name = args.model.name

# Check for optional fields
if hasattr(args.model, 'checkpoint_path'):
    ...
```

## Checkpointing

### Checkpoint Frequency

```bash
# Save checkpoint every 10,000 steps
python -m nanoT5.main checkpoint.every_steps=10000

# Save only at the end (default for pre-training)
python -m nanoT5.main checkpoint.every_steps=100000
```

### Checkpoint Directory Structure

Checkpoints are saved in Hydra's output directory:

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/
        ├── .hydra/              # Hydra metadata
        │   ├── config.yaml      # Resolved config
        │   └── overrides.yaml   # CLI overrides
        ├── checkpoint-pt-10000/ # Checkpoint at step 10000
        │   ├── pytorch_model.bin
        │   ├── optimizer.bin
        │   ├── scheduler.bin
        │   └── random_states_*.pkl
        ├── checkpoint-pt-65536/ # Final checkpoint
        │   └── ...
        ├── tokenizer/           # Saved tokenizer
        │   ├── tokenizer.json
        │   └── ...
        └── config.yaml          # T5 model config
```

### Loading a Checkpoint

```bash
# Continue training from checkpoint
python -m nanoT5.main \
    model.checkpoint_path=/path/to/checkpoint-pt-10000/pytorch_model.bin

# Fine-tune from checkpoint
python -m nanoT5.main task=ft \
    model.checkpoint_path=/path/to/checkpoint-pt-65536/pytorch_model.bin
```

## Logging

### Console Logging

Training logs are printed every `logging.every_steps` steps:

```
[train] Step 100 out of 65536 | Loss --> 59.881 | Grad_l2 --> 61.126 | Weights_l2 --> 7042.931 | Lr --> 0.010 | Seconds_per_step --> 1.385 |
```

### WandB Logging

Enable WandB in `nanoT5/configs/default.yaml`:

```yaml
logging:
  use_wandb: true
  wandb_config:
    project: nanoT5
    entity: your_username
    tags: ['nanoT5', 'experiment_name']
    mode: 'online'  # or 'offline', 'disabled'
```

Or override via CLI:

```bash
python -m nanoT5.main \
    logging.use_wandb=true \
    logging.wandb_config.project=my_project \
    logging.wandb_config.entity=my_username
```

WandB logs:
- `train/loss`, `train/grad_l2`, `train/weights_l2`, `train/lr`, `train/seconds_per_step`
- `eval/loss`, `eval/accuracy`, `eval/time`
- `test/rougeL`, `test/time` (fine-tuning only)

### Logged Metrics

**Pre-training**:
- Loss (negative log-likelihood)
- Gradient L2 norm (if `logging.grad_l2=true`)
- Weights L2 norm (if `logging.weights_l2=true`)
- Learning rate
- Seconds per step

**Fine-tuning** (additional):
- Accuracy (token-level accuracy on decoder outputs)
- ROUGE-L (generation quality on test set)

## Limits and Edge Cases

### Sequence Length

Maximum input length is determined by `data.input_length` (default 512 for pre-training). For fine-tuning:
- `data.max_seq_len: 1024` - Max encoder input
- `data.max_target_len: 128` - Max decoder output

Longer sequences require more memory and may not fit on a single GPU.

### Streaming Dataset Limitations

When using streaming C4 (`mode=pt`):
- No `len(dataloader)` - dataset is infinite
- Training stops at `optim.total_steps` (not epochs)
- Shuffling is limited to buffer size (10,000 examples)

### Epoch-Based Training

For fine-tuning, you can use epoch-based training:

```bash
python -m nanoT5.main task=ft optim.epochs=3
```

**Note**: `optim.epochs` overwrites `optim.total_steps` if `epochs > 0`. Only works for non-streaming datasets.

### Config Validation

The code performs runtime validation (`gen_utils.py:11-24`):
- `batch_size % grad_acc == 0` - Must be divisible
- `eval.every_steps % logging.every_steps == 0` - Eval must align with logging
- `device=gpu` requires `torch.cuda.is_available()`
- `eval_only` and `predict_only` are mutually exclusive

If validation fails, the script will error with an assertion message.

### No-Ops and Unused Flags

- Setting `optim.grad_clip=0.0` disables gradient clipping (but still computes grad L2 norm if `logging.grad_l2=true`)
- Setting `logging.use_wandb=false` disables WandB (but still calls `wandb.log()` harmlessly)
- The `local_env` config composition point exists but has no default overrides

## Example: Full Pre-Training to Fine-Tuning Pipeline

```bash
# Step 1: Pre-train for 16 hours
python -m nanoT5.main task=pt_16h logging.use_wandb=true

# Note the output directory (e.g., outputs/2023-06-20/14-30-00)

# Step 2: Clone SNI dataset
git clone https://github.com/allenai/natural-instructions.git data

# Step 3: Fine-tune from the checkpoint
python -m nanoT5.main task=ft \
    model.klass=local_t5 \
    model.checkpoint_path=outputs/2023-06-20/14-30-00/checkpoint-pt-53332/pytorch_model.bin \
    logging.use_wandb=true

# Step 4: Evaluate
python -m nanoT5.main task=ft \
    predict_only=true \
    model.klass=local_t5 \
    model.checkpoint_path=outputs/2023-06-20/16-45-00/checkpoint-ft-XXXXX/pytorch_model.bin
```

This reproduces the core experiments from the nanoT5 paper.
