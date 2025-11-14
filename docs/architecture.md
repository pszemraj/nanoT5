# Architecture

This document describes the system architecture and code organization of nanoT5.

## Entry Point and Execution Flow

### Main Entry: `nanoT5/main.py`

The primary entry point is `main.py:23-86`, decorated with `@hydra.main` for configuration management. Execution proceeds as follows:

1. **Initialize Accelerator** (`main.py:25-28`): Creates HuggingFace Accelerate object for distributed training, mixed precision, and device placement
2. **Setup basics** (`main.py:29`): Logging, seed setting, environment checks, TF32 flags
3. **Load tokenizer** (`main.py:30`): Loads T5 tokenizer from HuggingFace or custom path
4. **Get config** (`main.py:32`): Creates T5Config, updates vocab size, special tokens
5. **Build model** (`main.py:33`): Instantiates either `MyT5` (local) or `T5ForConditionalGeneration` (HuggingFace)
6. **Create optimizer** (`main.py:34`): Instantiates AdamWScale, AdamW, or Adafactor
7. **Create LR scheduler** (`main.py:35`): Cosine, inverse-square-root (legacy), or constant
8. **Build dataloaders** (`main.py:36`): Constructs train/test dataloaders with streaming or static datasets
9. **Prepare with Accelerator** (`main.py:42-50`): Wraps model, optimizer, scheduler, dataloaders for multi-GPU/mixed-precision
10. **Optional compile** (`main.py:52-53`): Applies `torch.compile` if `model.compile=true`
11. **Execute mode** (`main.py:60-79`):
    - `eval_only`: Run evaluation loop only
    - `predict_only`: Run prediction loop only (fine-tuning mode)
    - Default: Run training loop
12. **Save tokenizer and finish** (`main.py:81-82`)

### Configuration System (Hydra)

Hydra manages hierarchical YAML configs with composition and CLI overrides.

**Config structure**:
```
nanoT5/configs/
├── default.yaml          # Base config (mode, device, precision, model, optim, eval, logging)
├── task/
│   ├── pt.yaml           # Pre-training defaults (empty, uses default.yaml)
│   ├── pt_4h.yaml        # 4-hour budget overrides
│   ├── pt_8h.yaml        # 8-hour budget overrides
│   ├── pt_12h.yaml       # 12-hour budget
│   ├── pt_16h.yaml       # 16-hour budget (recommended)
│   ├── pt_20h.yaml       # 20-hour budget
│   ├── pt_24h.yaml       # 24-hour budget
│   └── ft.yaml           # Fine-tuning config (SNI dataset, AdamW, constant LR)
└── local_env/
    └── default.yaml      # Environment-specific overrides (currently empty)
```

**Composition** (`default.yaml:1-4`):
```yaml
defaults:
    - _self_
    - task: pt
    - local_env: default
```

**CLI overrides**:
```bash
python -m nanoT5.main task=pt_16h optim.name=adamwscale optim.lr_scheduler=cosine
```

Hydra creates a timestamped output directory in `outputs/YYYY-MM-DD/HH-MM-SS/` where checkpoints, logs, and config snapshots are saved.

## Module Structure

### `nanoT5/utils/`

All core functionality lives in this package:

#### `model_utils.py` (448 lines)

Central hub for model and data construction:

- `get_model(args, config)` - Factory for `MyT5` or `T5ForConditionalGeneration`
- `get_config(args, tokenizer)` - Builds `T5Config`, updates vocab/special tokens, applies overrides
- `get_tokenizer(args)` - Loads tokenizer, validates T5 special tokens
- `load_dataset_splits(args)` - Loads C4 (streaming) or SNI (static) datasets
- `process_dataset(dataset_splits, args, tokenizer)` - Tokenizes, shuffles, applies span-masking logic
- `get_data_collator(tokenizer, config, args)` - Returns `DataCollatorForT5MLM` or `DataCollatorForNI`
- `get_dataloaders(tokenizer, config, args)` - Constructs PyTorch DataLoaders
- `get_optimizer(model, args)` - Factory for AdamW, AdamWScale, or Adafactor (with weight decay groups)
- `get_lr_scheduler(optimizer, args, logger)` - Creates cosine, inverse-sqrt, or constant scheduler
- `model_summary(model, max_depth=4)` - Prints model architecture with parameter counts

**Key pattern**: Pre-training (`mode=pt`) uses streaming C4 with `DataCollatorForT5MLM`; fine-tuning (`mode=ft`) uses static SNI with `DataCollatorForNI`.

#### `t5_model.py` (609 lines)

Custom PyTorch implementation of T5:

- `MyT5` - Main model class (encoder-decoder)
- `T5Stack` - Encoder or decoder stack (shared embedding, blocks, layer norm)
- `T5Block` - Single transformer block (self-attention, optional cross-attention, feedforward)
- `T5LayerSelfAttention` - Self-attention layer with residual and pre-norm
- `T5LayerCrossAttention` - Cross-attention layer for decoder
- `T5LayerFF` - Feedforward layer (gated activation)
- `T5Attention` - Multi-head attention with relative positional bias
- `EncoderOutput`, `Seq2SeqLMOutput` - Dataclasses for outputs

Simplified from HuggingFace but functionally equivalent. Uses T5's relative positional embeddings (not learned embeddings, not ALiBi). Implements greedy decoding in `generate()` (no beam search).

#### `train_utils.py` (223 lines)

Training loop helpers:

- `train(model, train_dataloader, test_dataloader, ...)` - Main training loop
  - Iterates epochs, handles gradient accumulation
  - Calls `forward()`, `maybe_logging()`, `maybe_eval_predict()`, `maybe_save_checkpoint()`
- `forward(model, batch, calc_acc=False)` - Forward pass, returns loss and stats
- `eval(model, dataloader, logger, args, tokenizer)` - Evaluation loop (loss + accuracy)
- `predict(model, dataloader, logger, args, tokenizer)` - Generation loop (ROUGE-L computation)
- `maybe_grad_clip_and_grad_calc(accelerator, model, args)` - Gradient clipping and L2 norm logging
- `extra_stats(args, model, optimizer)` - Collects LR, weight L2, time per step
- `maybe_*()` functions - Conditional execution based on step counters

**Training flow**:
1. Loop over epochs (or until `total_steps` reached)
2. For each batch: forward pass, backward pass (scaled by `grad_acc`), accumulate stats
3. Every `grad_acc` batches: clip gradients, optimizer step, scheduler step, zero gradients
4. Every `logging.every_steps`: log stats to console/WandB
5. Every `eval.every_steps`: evaluate on test set
6. Every `checkpoint.every_steps`: save checkpoint

#### `copied_utils.py` (629 lines)

Utilities copied/adapted from external sources:

- `DataCollatorForT5MLM` - Span-corruption masking for pre-training (from HuggingFace Flax example)
  - `random_spans_noise_mask()` - Generates random span masks
  - `create_sentinel_ids()` - Replaces spans with sentinel tokens
  - `filter_input_ids()` - Creates input/target sequences
- `compute_input_and_target_lengths()` - Calculates sequence lengths to avoid padding
- `AdamWScale` - AdamW with RMS-based LR scaling (core innovation of this repo)
  - `_rms(tensor)` - Computes root-mean-square of tensor
  - `step()` - Optimizer step with `step_size *= max(1e-3, rms(param))`
- `tokenize_function()` - Tokenizes and concatenates text for C4
- `DataCollatorForNI` - Formats Super-Natural Instructions examples with task definitions, pos/neg examples

#### `ni_dataset.py` (196 lines)

HuggingFace dataset loader for Super-Natural Instructions:

- `NaturalInstructions` - Dataset builder for SNI
- `_split_generators()` - Loads train/test task lists
- `_generate_examples()` - Reads task JSON files, yields instances

Expects directory structure:
```
data/
├── splits/default/
│   ├── train_tasks.txt
│   └── test_tasks.txt
└── tasks/
    └── task_*.json
```

#### `logging_utils.py` (93 lines)

Logging infrastructure:

- `Logger` - Main logger class
  - `setup_wandb()` - Initializes WandB if enabled
  - `log_stats()` - Logs to console and WandB
  - `log_message()` - Console logging
- `Averager` - Running averager for training stats

#### `gen_utils.py` (70 lines)

General utilities:

- `setup_basics()` - Initializes logging, seeds, checks environment
- `check_args_and_env()` - Validates config (batch size divisibility, device availability)
- `opti_flags()` - Enables TF32, sets BF16 flag on config
- `update_args_with_env_info()` - Adds SLURM job ID, working dir to config
- `update_paths()` - Converts relative paths to absolute for fine-tuning

## Data Flow

### Pre-Training (mode=pt)

```
C4 dataset (streaming)
  ↓ load_dataset("c4", "en", streaming=True)
  ↓ remove_columns(["timestamp", "url"])
  ↓ tokenize_function (concatenate, chunk to before_mask_input_length)
  ↓ shuffle(buffer_size=10_000)
  ↓ DataLoader (num_workers=8, batch_size=batch_size/grad_acc)
  ↓ DataCollatorForT5MLM (span-corruption masking)
  ↓ Model forward (encoder-decoder)
  ↓ Loss (CrossEntropyLoss on decoder outputs)
  ↓ Backward, gradient accumulation
  ↓ Optimizer step (AdamWScale + cosine LR)
```

**Key insight**: C4 is never fully downloaded. Streaming + on-the-fly preprocessing + shuffling happens in parallel with training. No separate preprocessing step required.

### Fine-Tuning (mode=ft)

```
Super-Natural Instructions dataset (static)
  ↓ load_dataset(ni_dataset.py, data_dir, task_dir)
  ↓ Read task JSONs, create instances
  ↓ DataLoader (shuffle train, no shuffle test)
  ↓ DataCollatorForNI (format with definition, examples, input)
  ↓ Model forward (encoder-decoder)
  ↓ Loss (CrossEntropyLoss on decoder outputs)
  ↓ Generation (greedy decoding for ROUGE-L)
```

## Model Construction Paths

### Path 1: Local T5 from scratch (default for pre-training)

```
args.model.klass = "local_t5"
args.model.random_init = true
  ↓ get_config(args, tokenizer) → T5Config from "google/t5-v1_1-base"
  ↓ get_model(args, config) → MyT5(config)
  ↓ MyT5.__init__() → random initialization via _init_weights()
```

### Path 2: Local T5 from checkpoint

```
args.model.klass = "local_t5"
args.model.checkpoint_path = "/path/to/pytorch_model.bin"
  ↓ get_config(args, tokenizer) → T5Config
  ↓ get_model(args, config) → MyT5(config)
  ↓ model.load_state_dict(torch.load(checkpoint_path))
```

### Path 3: HuggingFace T5 from Hub (default for fine-tuning)

```
args.model.klass = "hf_t5"
args.model.random_init = false
args.model.checkpoint_path = ""
  ↓ get_config(args, tokenizer) → T5Config
  ↓ get_model(args, config) → T5ForConditionalGeneration.from_pretrained(args.model.name, config=config)
```

## Abstraction Patterns

### 1. Mode-based dispatch

Many functions branch on `args.mode` (either "pt" or "ft"):
- `load_dataset_splits()` - C4 vs. SNI
- `process_dataset()` - Span masking vs. raw text
- `get_data_collator()` - `DataCollatorForT5MLM` vs. `DataCollatorForNI`

### 2. Factory functions

All major components use factory functions in `model_utils.py`:
- `get_model()` - Selects model class based on `args.model.klass`
- `get_optimizer()` - Selects optimizer based on `args.optim.name`
- `get_lr_scheduler()` - Selects scheduler based on `args.optim.lr_scheduler`

### 3. Accelerate-wrapped training

HuggingFace Accelerate handles:
- Mixed precision (BF16/TF32)
- Device placement
- Gradient accumulation (implicitly via scaled backward)
- Gradient clipping
- Checkpoint saving/loading

Pattern: All objects (`model`, `optimizer`, `scheduler`, `dataloaders`) are wrapped via `accelerator.prepare()` before use.

### 4. Hydra config with OmegaConf

Configs are nested `OmegaConf.DictConfig` objects. Runtime modifications use `open_dict()` context manager:

```python
with open_dict(args):
    args.current_train_step = 1
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
```

## Key Design Decisions

### Streaming C4
Pre-training uses `datasets.load_dataset("c4", "en", streaming=True)` which returns an `IterableDataset`. This avoids downloading 300GB of data before training starts. Trade-off: No len(), no random access, requires `shuffle(buffer_size=10_000)` for randomness.

### Gradient Accumulation
Micro-batch size is **derived** as `batch_size / grad_acc`, not specified. This differs from HuggingFace Trainer where micro-batch size is primary. Accumulation is manual via `loss / grad_acc` before backward pass (`train_utils.py:202`).

### Simplified T5
`MyT5` removes features from HuggingFace implementation:
- No caching of past key-values (no KV-cache)
- No beam search (greedy decoding only)
- No head masking, output attentions, output hidden states
- No gradient checkpointing
- No encoder-decoder attention cache

Trade-off: Cleaner code, easier to understand, but slower inference and limited decoding options.

### AdamWScale
RMS scaling (`step_size *= max(1e-3, rms(param))`) is applied per-parameter, adapting learning rate to parameter magnitude. This is the key difference from standard AdamW and enables better convergence than AdamW alone. Borrowed from Adafactor's adaptive step size.
