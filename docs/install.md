# Installation & Environment Setup

This document describes how to set up the environment to run nanoT5.

## System Requirements

### Hardware

- **Recommended**: NVIDIA GPU with 40GB+ VRAM (e.g., A100)
- **Minimum for pre-training**: 24GB VRAM (requires increasing gradient accumulation steps)
- **CPU-only**: Supported but significantly slower

The code is GPU-optimized but will run on CPU if no CUDA device is available.

### Software

- Python 3.10+
- CUDA-compatible GPU drivers (if using GPU)
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/PiotrNawrot/nanoT5.git
cd nanoT5
```

### 2. Create Python Environment

Using conda (recommended):

```bash
conda create -n nanoT5 python=3.10 -y
conda activate nanoT5
```

Or using venv:

```bash
python3.10 -m venv nanoT5_env
source nanoT5_env/bin/activate  # On Windows: nanoT5_env\Scripts\activate
```

### 3. Install PyTorch

#### For CPU-only installations:

```bash
pip install ninja
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### For GPU installations:

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the installation command for your CUDA version. Typically:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

**Note**: PyTorch 2.0.1 or higher is required for `torch.compile` support.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `accelerate` - Multi-GPU and mixed-precision training
- `datasets >= 1.8.0` - HuggingFace datasets (C4, streaming support)
- `transformers` - HuggingFace transformers (model configs, tokenizers, optimizers)
- `sentencepiece != 0.1.92` - Tokenization (version 0.1.92 has known issues)
- `hydra-core` - Configuration management
- `wandb` - Experiment tracking (optional, can be disabled)
- `evaluate` - Metrics computation
- `nltk` - Text processing for evaluation
- `rouge_score` - ROUGE metric for fine-tuning evaluation
- `absl-py` - Utilities
- `pyyaml` - YAML config parsing
- `pynvml` - GPU monitoring
- `protobuf==3.20.*` - Protocol buffers (pinned for compatibility)
- `pdbpp` - Enhanced debugger
- `notebook` - Jupyter notebook support

### 5. Verify Installation

Check that PyTorch can access your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

For CPU-only installations, ensure PyTorch is installed:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Environment Configuration

### Mixed Precision Support

nanoT5 supports multiple precision formats:

- **BF16** (bfloat16): Recommended for A100 GPUs. Set `precision: 'bf16'` in config.
- **TF32** (TensorFloat-32): Automatically enabled on Ampere GPUs (A100, etc.) for matmul operations.
- **FP16** (float16): Not recommended - training diverges in most experiments.
- **FP32** (float32): Slowest but most stable. Set `precision: 'no'` in config.

### PyTorch 2.0 Compile

PyTorch 2.0's `torch.compile` provides significant speedups. It is enabled by default via `model.compile: true` in the configuration. If you encounter issues, disable it by setting `model.compile: false`.

### Weights & Biases (Optional)

WandB is used for experiment tracking. To use it:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Log in: `wandb login`
3. Configure in `nanoT5/configs/default.yaml`:
   ```yaml
   logging:
     use_wandb: true
     wandb_config:
       project: nanoT5
       entity: 'your_wandb_username'
       tags: ['nanoT5', 'my_tag']
       mode: 'online'
   ```

To disable WandB, set `logging.use_wandb: false` in the config.

## Fine-Tuning Dataset Setup (Optional)

To fine-tune on Super-Natural Instructions:

```bash
git clone https://github.com/allenai/natural-instructions.git data
```

This creates a `data/` directory with task definitions and train/test splits. The fine-tuning config (`task=ft`) expects:
- `data/splits/default/train_tasks.txt`
- `data/splits/default/test_tasks.txt`
- `data/tasks/*.json` (task definition files)

## Known Issues

### Compatibility

- **sentencepiece 0.1.92** is known to cause issues - it is excluded in requirements.txt
- **protobuf 3.20.*** is required for compatibility with transformers/datasets
- **FP16 precision** causes training divergence - use BF16 or TF32 instead

### GPU Memory

If you encounter OOM (out-of-memory) errors during pre-training:

1. Increase `optim.grad_acc` (gradient accumulation steps)
2. Ensure `optim.batch_size` is divisible by `optim.grad_acc`
3. The micro batch size is `batch_size / grad_acc`

Example: For a 24GB GPU, use `optim.grad_acc: 2` instead of the default `1`.

### CPU Training

CPU training is supported but extremely slow (74+ hours for full pre-training vs. 10 hours on A100 with BF16+compile). Only recommended for testing or small experiments.
