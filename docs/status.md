# Project Status and Limitations

This document provides an honest assessment of the codebase's maturity, known issues, and limitations.

## Maturity by Component

### Stable and Production-Ready

**Pre-training pipeline** (`mode=pt`):
- ✅ Streaming C4 dataset loading and preprocessing
- ✅ Span-corruption masking (T5 objective)
- ✅ AdamWScale optimizer with cosine LR schedule
- ✅ Mixed-precision training (BF16, TF32)
- ✅ PyTorch 2.0 compile support
- ✅ Gradient accumulation
- ✅ Logging (console + WandB)
- ✅ Checkpointing with Accelerate

**Status**: Extensively tested, reproduces published results, ready for research use.

**Fine-tuning pipeline** (`mode=ft`):
- ✅ Super-Natural Instructions dataset loader
- ✅ Task definition + example formatting
- ✅ Standard AdamW optimizer with constant LR
- ✅ ROUGE-L evaluation
- ✅ Generation (greedy decoding)

**Status**: Works reliably, reproduces Tk-Instruct baseline. Limited to SNI dataset format.

**Custom T5 model** (`MyT5`):
- ✅ Encoder-decoder architecture
- ✅ Relative positional embeddings
- ✅ Gated activation (SiLU)
- ✅ Pre-layer normalization
- ✅ Greedy generation

**Status**: Simplified but correct implementation. Suitable for training and basic inference. Missing advanced features (beam search, KV-cache).

### Experimental / Incomplete

**Evaluation modes**:
- ⚠️ `eval_only=true` - Works but only computes loss/accuracy
- ⚠️ `predict_only=true` - Only available for fine-tuning (`mode=ft`)

**Status**: Limited functionality. No standalone inference script. Must use training script with flags.

**Local environment overrides** (`local_env/*.yaml`):
- ⚠️ Config composition point exists but default file is empty
- No environment-specific customization implemented

**Status**: Placeholder for future use. Currently unused.

### Not Implemented

The following features are explicitly **not present** in this codebase:

- ❌ **Multi-GPU training**: No DistributedDataParallel, no DeepSpeed, no FSDP
- ❌ **Model parallelism**: No tensor parallelism, no pipeline parallelism
- ❌ **Beam search**: Generation only supports greedy decoding
- ❌ **KV-cache**: No caching of past key-values during generation (slower inference)
- ❌ **Flash Attention**: Uses standard PyTorch attention (not memory-efficient attention)
- ❌ **Gradient checkpointing**: No activation checkpointing for memory savings
- ❌ **Alternative positional embeddings**: ALiBi was attempted but didn't work
- ❌ **FP16 support**: Training diverges with FP16 in all tested configs
- ❌ **Inference server**: No deployment code, no FastAPI/gRPC endpoints
- ❌ **Quantization**: No int8/int4 quantization for deployment
- ❌ **ONNX export**: No export to ONNX or other deployment formats
- ❌ **Resume training from checkpoint**: Can load model weights but not optimizer/scheduler state
- ❌ **Custom tokenizers**: Only HuggingFace-compatible tokenizers with T5 special tokens
- ❌ **Non-English support**: C4 is English-only, no multilingual pre-training
- ❌ **Other pre-training objectives**: Only span-corruption (no prefix LM, causal LM, etc.)
- ❌ **Validation split for C4**: Uses only train and test (no dev set)

## Known Issues and Limitations

### 1. FP16 Training Divergence

**Issue**: Training with `precision=fp16` diverges across all tested configurations, optimizers, and learning rates.

**Affected**: Pre-training and fine-tuning.

**Workaround**: Use `precision=bf16` (A100/H100) or `precision=tf32` (Ampere GPUs) or `precision=no` (FP32).

**Root cause**: Unknown. Likely gradient underflow or loss of precision in embeddings/layer norms.

**Code location**: Precision is set in `main.py:25-28` via Accelerator.

### 2. Sequence Length Constraints

**Issue**: Maximum sequence length is constrained by memory. Default is 512 (encoder) for pre-training.

**Affected**: Long documents, long-context tasks.

**Limitation**: Increasing `data.input_length` beyond 512 may cause OOM on 40GB/80GB GPUs.

**Workaround**: Reduce batch size or increase gradient accumulation. For 1024 tokens, use `optim.batch_size=64` instead of 128.

**Code location**: `data.input_length` in `default.yaml:26`.

### 3. Streaming Dataset Limitations

**Issue**: C4 is an `IterableDataset` with no `len()` and no random access.

**Affected**: Pre-training only.

**Implications**:
- Cannot compute epochs (must use fixed `total_steps`)
- Shuffling is limited to buffer size (10,000 examples)
- Cannot checkpoint at exact data positions (may see duplicates if resuming)

**Workaround**: None. This is inherent to streaming datasets.

**Code location**: `load_dataset_splits()` in `model_utils.py:106-135`.

### 4. Gradient Accumulation Constraints

**Issue**: `optim.batch_size` must be divisible by `optim.grad_acc`. Micro-batch size is derived, not specified.

**Affected**: All training.

**Example error**: Setting `batch_size=100 grad_acc=3` will error because 100 / 3 is not an integer.

**Workaround**: Adjust values to be divisible. E.g., `batch_size=96 grad_acc=3` → micro_batch=32.

**Code location**: Validated in `check_args_and_env()` in `gen_utils.py:12`.

### 5. WandB Logging Bugs

**Issue**: If `logging.use_wandb=false`, the code still calls `wandb.log()` (harmless but may print warnings).

**Affected**: Logging only.

**Severity**: Low. Does not affect training, only console output.

**Code location**: `logger.log_stats()` in `logging_utils.py:74-76` checks `wandb.run` but not `use_wandb` flag.

### 6. Checkpoint Resume Incomplete

**Issue**: `model.checkpoint_path` loads model weights but not optimizer/scheduler state. Training restarts from step 1.

**Affected**: Resuming interrupted training.

**Workaround**: Accelerate saves full state in `checkpoint-*` directories. To resume, use `accelerator.load_state()` (not currently wired in `main.py`).

**Status**: Feature exists in Accelerate but not exposed in this codebase.

### 7. No Validation During Pre-Training

**Issue**: Evaluation happens at `eval.every_steps=100000` (end of training) by default. No intermediate validation.

**Affected**: Pre-training only.

**Implication**: Cannot detect overfitting or divergence during training.

**Workaround**: Set `eval.every_steps=5000` (or smaller) to evaluate more frequently.

**Code location**: `default.yaml:45`.

### 8. Generation Speed

**Issue**: Greedy decoding is slow compared to HuggingFace's implementation.

**Reason**: No KV-cache (recomputes encoder outputs and past key-values every step).

**Affected**: `predict_only=true` mode and fine-tuning evaluation.

**Performance**: ~1.3x slower than HuggingFace for 20 input/output pairs (9.8s vs 11.4s reported in code comment at `t5_model.py:486`).

**Workaround**: Use HuggingFace T5 (`model.klass=hf_t5`) for production inference.

### 9. SNI Dataset Path Hardcoded

**Issue**: Fine-tuning expects `data/` directory with specific structure (`data/splits/default/`, `data/tasks/`).

**Affected**: Fine-tuning only.

**Limitation**: Cannot easily use other datasets without modifying `ni_dataset.py` or config paths.

**Workaround**: Clone SNI to `data/` or override paths in config:
```bash
python -m nanoT5.main task=ft data.data_dir=/custom/path/splits data.task_dir=/custom/path/tasks
```

**Code location**: `ft.yaml:20-21`.

## Technical Debt

### 1. Manual Gradient Accumulation

**What**: Gradient accumulation is implemented manually in `train_utils.py:175-222` instead of using Accelerate's built-in support.

**Why it exists**: Likely written before Accelerate supported gradient accumulation, or for explicit control.

**Impact**: More complex training loop. Harder to maintain.

**Fix**: Use `accelerator.accumulate(model)` context manager (requires refactoring loop).

### 2. Hardcoded Special Tokens

**What**: Code assumes T5 tokenizer with 100 `<extra_id_*>` tokens (`model_utils.py:97-101`).

**Why it exists**: T5-specific implementation.

**Impact**: Cannot use non-T5 tokenizers (e.g., BERT, GPT-2).

**Fix**: Make sentinel token pattern configurable or detect from tokenizer vocab.

### 3. Config Mutation with open_dict

**What**: Runtime config modification uses `open_dict()` context manager in multiple places.

**Example**: `main.py:55-58`, `model_utils.py:46-47`, etc.

**Why it exists**: OmegaConf configs are read-only by default.

**Impact**: Config changes are scattered across codebase, hard to track.

**Fix**: Centralize config updates or use mutable configs.

### 4. Averager Reset Timing

**What**: `Averager` resets after computing average (`logging_utils.py:30`), not before update.

**Why it exists**: Stateful design.

**Impact**: Subtle bugs if average is called multiple times without updates.

**Fix**: Separate reset from average or make stateless.

### 5. Magic Numbers

**What**: Several hardcoded values without explanation:
- Buffer size 10,000 for shuffling (`model_utils.py:166`)
- Max depth 4 for model summary (`model_utils.py:377`)
- 0.9 split for inverse-sqrt LR schedule (`model_utils.py:336`)

**Why it exists**: Ported from original T5 or empirically chosen.

**Impact**: Hard to customize without code changes.

**Fix**: Move to config with defaults.

### 6. Unused Imports and Code

**What**: Some imports are unused:
- `datasets.iterable_dataset.IterableDataset` is imported but only for type checking

**Why it exists**: Leftover from development.

**Impact**: Minimal (slightly slower import time).

**Fix**: Remove unused imports.

### 7. Global WandB State

**What**: WandB is initialized in `Logger` but accessed globally via `wandb.log()` and `wandb.finish()`.

**Why it exists**: WandB's global state design.

**Impact**: Difficult to test, hard to use multiple loggers.

**Fix**: Encapsulate WandB calls in Logger class.

## Experimental Features Tried and Abandoned

### ALiBi Positional Embeddings

**What**: Attempted to replace T5's relative positional embeddings with ALiBi (Attention with Linear Biases).

**Goal**: Reduce parameter count, enable Flash Attention, improve length generalization.

**Result**: Training was less stable and pre-training loss was worse.

**Conclusion**: Abandoned. T5's learned relative bias works better for this model size and dataset.

**Evidence**: Mentioned in main README.md:180.

### Lion Optimizer

**What**: Tested Lion optimizer (recently published, claims to be more memory-efficient than AdamW).

**Result**: Did not outperform AdamWScale in pre-training loss.

**Conclusion**: Abandoned.

**Evidence**: Mentioned in main README.md:179.

### Sophia Optimizer

**What**: Tested Sophia optimizer (second-order method).

**Result**: Did not outperform AdamWScale.

**Conclusion**: Abandoned.

**Evidence**: Mentioned in main README.md:179.

## Dragons (Fragile or Risky Code)

### 1. Vocab Size Rounding

**Location**: `model_utils.py:56`

```python
config.vocab_size = math.ceil(len(tokenizer) / 128) * 128
```

**Why**: Ensures vocab size is divisible by 128 for efficient GPU operations.

**Risk**: If tokenizer has non-standard vocab size, this may add unexpected padding tokens.

**Impact**: Model will allocate embedding weights for unused tokens (slight memory waste).

### 2. Streaming Dataset Shuffle Buffer

**Location**: `model_utils.py:166`

```python
dataset_split = dataset_split.shuffle(buffer_size=10_000, seed=args.seed)
```

**Why**: Streaming datasets require finite buffer for shuffling.

**Risk**: If buffer size is too small, training data is not truly randomized (quasi-sequential).

**Impact**: May affect convergence or generalization. 10,000 is reasonable for C4 but not validated.

### 3. Loss Scaling for Gradient Accumulation

**Location**: `train_utils.py:202`

```python
accelerator.backward(loss / args.optim.grad_acc)
```

**Why**: Gradients must be scaled by 1/grad_acc when accumulating.

**Risk**: If grad_acc is changed without understanding this, gradients will have wrong magnitude.

**Impact**: Training divergence or incorrect convergence.

### 4. Sentinel Token Indexing

**Location**: `copied_utils.py:100-103`

```python
sentinel_ids = np.where(sentinel_ids != 0, (len(tokenizer) - sentinel_ids), 0)
```

**Why**: T5 sentinel tokens are stored at end of vocab in reverse order.

**Risk**: Assumes specific tokenizer layout. If tokenizer is changed, this will break.

**Impact**: Incorrect span-corruption masks, garbled training data.

### 5. BF16 Embedding Cast

**Location**: `t5_model.py:372-373`

```python
if hasattr(config, "is_bf16") and config.is_bf16:
    inputs_embeds = inputs_embeds.to(torch.bfloat16)
```

**Why**: Accelerate doesn't autocast embeddings, must cast manually.

**Risk**: This only applies to `MyT5`. If using `hf_t5`, embeddings may be in wrong dtype.

**Impact**: Mixed-precision training may fail silently or have degraded performance.

### 6. EOS Masking in Generation

**Location**: `t5_model.py:508-513`

```python
mask = torch.arange(L, device=labels.device).unsqueeze(0) <= (labels == 1).long().argmax(-1).unsqueeze(-1)
labels = labels.masked_fill(~mask, 0)
```

**Why**: Masks out tokens after first EOS token (T5 uses PAD=0).

**Risk**: Complex indexing logic. If EOS token ID changes, this will break.

**Impact**: Generation will produce incorrect outputs (padding in middle of sequence).

## Scaling Limitations

### Single-GPU Only

**Limitation**: No multi-GPU support. Cannot train models larger than fit on a single GPU.

**Workaround**: Use gradient accumulation to simulate larger batch sizes, but this doesn't help with model size.

**To scale**: Would need to implement DistributedDataParallel (data parallelism) or DeepSpeed/FSDP (model parallelism).

### Memory Constraints

**Limitation**: T5-base (248M params) with BF16 and batch_size=128 requires ~40GB VRAM.

**Scaling up**: T5-large (770M) or T5-XL (3B) would require:
- Larger GPU (A100 80GB) or
- Gradient accumulation (slower training) or
- Model parallelism (not implemented)

**Scaling down**: T5-small (60M) works on 16GB GPUs but is not extensively tested.

### C4 Download Dependency

**Limitation**: Streaming requires active internet connection. If connection drops, training pauses.

**Workaround**: Download C4 once and use local files (requires modifying `load_dataset_splits()`).

**Impact**: Training in offline environments (e.g., air-gapped clusters) requires pre-download.

## Recommendations for Production Use

**Do**:
- ✅ Use this codebase for research experiments on limited compute
- ✅ Use it as a reference implementation for understanding T5 architecture
- ✅ Use AdamWScale if you're implementing T5 pre-training elsewhere
- ✅ Use the span-corruption data collator for custom pre-training tasks

**Don't**:
- ❌ Deploy `MyT5` for production inference (use HuggingFace T5 instead)
- ❌ Expect multi-GPU/multi-node scaling without significant refactoring
- ❌ Use this for training models larger than T5-base without modifications
- ❌ Rely on FP16 precision (it doesn't work)

**If you need**:
- **Production inference**: Use HuggingFace Transformers with `T5ForConditionalGeneration`
- **Multi-GPU training**: Use HuggingFace Trainer with DeepSpeed/FSDP
- **Larger models**: Use model parallelism frameworks (Megatron-LM, DeepSpeed)
- **Advanced generation**: Use HuggingFace's `.generate()` with beam search, sampling, etc.

## Summary

nanoT5 is a **research-grade pre-training pipeline** that successfully demonstrates:
- T5 pre-training is feasible on single GPUs in PyTorch
- AdamWScale improves convergence over standard AdamW
- Streaming data preprocessing is viable for large datasets

However, it is **not a production system**. It lacks:
- Multi-GPU support
- Robust inference capabilities
- Advanced generation features
- Deployment tooling

Use it for research, learning, and small-scale experimentation. For production, integrate the insights (AdamWScale, streaming data) into a more robust framework like HuggingFace Trainer or Megatron-LM.
