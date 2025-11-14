# nanoT5 Documentation

nanoT5 is a PyTorch implementation for pre-training and fine-tuning T5 (encoder-decoder transformer) models under limited compute budgets. It enables pre-training a 248M parameter T5-base model on a single A100 GPU in under 24 hours, achieving competitive performance with models trained on 150x more data.

## What This Project Is

This is a research-grade training pipeline for T5 models, optimized for limited compute environments. It addresses the gap between requiring massive compute infrastructure (thousands of TPUs) and wanting to pre-train or experiment with encoder-decoder language models from scratch.

### Primary Components

- **Custom T5 model implementation** (`nanoT5/utils/t5_model.py`): Simplified, PyTorch-native T5 encoder-decoder architecture
- **Streaming data pipeline**: C4 dataset preprocessing happens in parallel with training (no separate preprocessing step required)
- **Pre-training loop**: T5 span-corruption objective with streaming data and gradient accumulation
- **Fine-tuning pipeline**: Super-Natural Instructions (SNI) dataset loader and training loop
- **Optimized optimizer**: AdamWScale - AdamW augmented with RMS scaling from Adafactor for better convergence
- **Configuration system**: Hydra-based configs for pre-training budgets (4h, 8h, 12h, 16h, 20h, 24h) and tasks (pre-training vs. fine-tuning)
- **Mixed-precision training**: BF16 and TF32 support with accelerate
- **Logging and tracking**: WandB integration, checkpointing, evaluation

### What's Not Here

- **No distributed training**: Single-GPU only; no model parallelism, tensor parallelism, or multi-node support
- **No inference server**: Training and evaluation only; no deployment code
- **No custom attention optimizations**: No Flash Attention, no custom CUDA kernels
- **No alternative positional embeddings**: Attempted ALiBi but it didn't work; uses standard T5 relative positional embeddings
- **No non-English support**: Focused on C4 English dataset
- **No evaluation harness beyond SNI**: Only tests on Super-Natural Instructions for downstream performance

## Why It Matters

This implementation demonstrates that **encoder-decoder pre-training is feasible on consumer/lab-grade hardware** in PyTorch:

- **Accessibility**: Enables researchers without TPU access to experiment with T5-style models
- **Reproducibility**: First PyTorch reproduction of T5 v1.1 pre-training (original is JAX/Flax)
- **Transparency**: Everything except the model architecture is exposed and optimized - data pipeline, optimizer, scheduler, training loop
- **Educational**: Simplified T5 implementation suitable for understanding encoder-decoder architectures

### Key Divergences from Standard Approaches

1. **Streaming data processing**: C4 is streamed and preprocessed on-the-fly rather than materialized to disk
2. **AdamWScale optimizer**: Augments AdamW with RMS-based learning rate scaling (borrowed from Adafactor) for better stability and convergence
3. **Cosine learning rate schedule**: Achieves better pre-training loss (1.953 NLL) compared to T5's original inverse-square-root schedule (1.995 NLL with Adafactor)
4. **PyTorch 2.0 compile**: Uses `torch.compile` for ~2x speedup on pre-training step time
5. **Explicit gradient accumulation**: Micro-batch size is derived from `batch_size / grad_acc`, not specified directly

## Performance Context

Achieves **40.7 ROUGE-L on SNI test set** after 16 hours of pre-training on a single A100, compared to 40.9 ROUGE-L for the original T5-base-v1.1 (pre-trained on 1 million steps × batch size 2048 on TPUs). This demonstrates that limited-budget pre-training can approach the performance of massively-scaled training.

Pre-training efficiency:
- **10.2 hours** (65,536 steps) with BF16 + torch.compile on A100 80GB
- **17.3 hours** with TF32 + torch.compile
- **23.7 hours** with BF16, no compile
- **74.6 hours** with FP32, no compile

## Documentation Structure

- **[install.md](install.md)** - Installation, dependencies, environment setup
- **[architecture.md](architecture.md)** - System design, code organization, execution paths
- **[usage.md](usage.md)** - How to run pre-training, fine-tuning, and evaluation
- **[internals.md](internals.md)** - Deep dive into T5 model, AdamWScale, data collators, and training mechanics
- **[status.md](status.md)** - Maturity assessment, known issues, experimental features, technical debt

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pre-train for 16 hours (recommended budget)
python -m nanoT5.main task=pt_16h

# Or use default config and override
python -m nanoT5.main optim.total_steps=10000

# Fine-tune on Super-Natural Instructions
python -m nanoT5.main task=ft model.checkpoint_path=/path/to/pytorch_model.bin
```

See [usage.md](usage.md) for detailed examples.

## Target Audience

This codebase is for:

- **Researchers** with limited compute who want to pre-train encoder-decoder models or test new pre-training ideas
- **ML engineers** who need to pre-train on custom/domain-specific corpora at small scale
- **Students/educators** learning about T5, encoder-decoder architectures, and language model pre-training
- **Anyone** experimenting with continued pre-training or instruction fine-tuning starting from scratch

If you just need a pre-trained T5 model for downstream tasks, **use HuggingFace Hub models instead**. The models trained here are worse than official checkpoints because they use 150x less compute.

## Limitations

- Single-GPU only
- No support for sequence lengths beyond 512 tokens (encoder) due to memory constraints
- FP16 training diverges in all tested configurations
- No production deployment features
- Limited to HuggingFace-compatible T5 tokenizers (requires T5 special tokens like `<extra_id_0>`)

See [status.md](status.md) for comprehensive limitations and technical debt.
