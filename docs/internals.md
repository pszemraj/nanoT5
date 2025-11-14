# Internals and Technical Details

This document explains the key technical components and algorithms in nanoT5.

## T5 Model Implementation

### Architecture Overview

The `MyT5` class (`t5_model.py:447-609`) implements the full T5 encoder-decoder architecture:

```
Input Text
  ↓ Shared Embedding (vocab_size × d_model)
  ↓ Encoder (T5Stack with is_decoder=False)
  ↓   → N encoder blocks (self-attention + feedforward)
  ↓ Hidden States (batch_size × seq_len × d_model)
  ↓ Decoder (T5Stack with is_decoder=True)
      → N decoder blocks (self-attention + cross-attention + feedforward)
  ↓ Final Layer Norm
  ↓ LM Head (d_model × vocab_size, no bias, unshared with embedding)
  ↓ Logits (batch_size × seq_len × vocab_size)
```

### Relative Positional Embeddings

T5 uses **learned relative positional bias** instead of absolute positional embeddings or sinusoidal encodings. Implementation in `T5Attention.compute_bias()` (`t5_model.py:123-148`):

1. Compute relative positions: `rel_pos[i,j] = j - i` (memory position - query position)
2. Bucket positions using logarithmic bucketing (`_relative_position_bucket()`, `t5_model.py:69-121`):
   - Bidirectional (encoder): 32 buckets split into 16 for positive, 16 for negative
   - Causal (decoder): All positions use only forward direction
   - First 16 buckets: exact positions 0, 1, 2, ..., 15
   - Remaining 16 buckets: logarithmically spaced from 16 to max_distance (128)
3. Look up bias values from learned embedding: `relative_attention_bias[bucket_id]` → (num_heads,)
4. Reshape to (1, num_heads, query_len, key_len) and add to attention scores

**Key property**: Position bias is computed only once in the first block, then reused in all subsequent blocks via pass-through (`t5_model.py:434-436`).

**Why this works**: Relative positions are translation-invariant and allow the model to generalize better to different sequence lengths. Bucketing reduces parameter count while maintaining expressiveness.

### Attention Mechanism

Standard multi-head attention with relative bias (`T5Attention.forward()`, `t5_model.py:150-231`):

```python
# Project to Q, K, V
query_states = q(hidden_states)  # (batch, seq_len, inner_dim)
key_states = k(hidden_states)    # (batch, key_len, inner_dim)
value_states = v(hidden_states)  # (batch, key_len, inner_dim)

# Reshape to (batch, num_heads, seq_len, d_kv)
query_states = query_states.view(batch, -1, num_heads, d_kv).transpose(1, 2)
key_states = key_states.view(batch, -1, num_heads, d_kv).transpose(1, 2)
value_states = value_states.view(batch, -1, num_heads, d_kv).transpose(1, 2)

# Compute attention scores
scores = matmul(query_states, key_states.transpose(-1, -2))  # (batch, heads, seq_len, key_len)

# Add relative position bias
scores += position_bias  # Broadcast addition

# Softmax and dropout
attn_weights = softmax(scores, dim=-1)
attn_weights = dropout(attn_weights, p=dropout_rate)

# Apply attention to values
attn_output = matmul(attn_weights, value_states)  # (batch, heads, seq_len, d_kv)

# Reshape and project
attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, inner_dim)
attn_output = o(attn_output)  # (batch, seq_len, d_model)
```

**No Flash Attention**: This is standard PyTorch attention. Flash Attention is not implemented because it requires non-parametric bias (T5 bias is learned).

### Pre-Norm Architecture

T5 uses pre-layer normalization (`T5LayerSelfAttention`, `t5_model.py:243-257`):

```python
normed = layer_norm(hidden_states)
attn_output = self_attention(normed)
hidden_states = hidden_states + dropout(attn_output)  # Residual connection
```

**Pattern**: Norm → Sublayer → Residual. This differs from post-norm (Sublayer → Residual → Norm) used in original Transformer.

**Why pre-norm**: More stable training, allows higher learning rates, no need for learning rate warmup (though we use warmup anyway).

### Gated Activation

T5 uses gated SiLU (also called gated-GELU in T5 v1.1) in the feedforward layer. The HuggingFace implementation (`T5DenseGatedActDense`, imported from transformers) computes:

```python
hidden_gelu = gelu(wi_0(x))  # First projection with GELU
hidden_linear = wi_1(x)      # Second projection (linear)
hidden = hidden_gelu * hidden_linear  # Element-wise multiplication (gating)
output = wo(dropout(hidden))  # Output projection
```

This is more expressive than standard ReLU/GELU feedforward.

### Generation (Greedy Decoding)

`MyT5.generate()` (`t5_model.py:471-515`) implements simple greedy decoding:

```python
labels = [0]  # Start with decoder_start_token_id (PAD)
encoder_outputs = None

for step in range(max_length):
    out = forward(input_ids, attention_mask, decoder_input_ids=labels, encoder_outputs=encoder_outputs)
    encoder_outputs = out.encoder_outputs  # Cache encoder outputs

    top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)  # Greedy selection
    labels = cat([labels, top_labels], dim=-1)

    if all sequences have produced EOS (token 1):
        break

# Force EOS at end, mask out padding after first EOS
```

**No beam search**: Only greedy decoding is implemented. Beam search would require tracking multiple hypotheses and is not needed for the experiments in this repo.

**Encoder caching**: Encoder outputs are computed once and reused for all decoding steps. This is critical for efficiency.

## AdamWScale Optimizer

The key innovation in this repo is `AdamWScale` (`copied_utils.py:249-375`), which augments AdamW with **RMS-based learning rate scaling**.

### Standard AdamW Step

```python
# Compute first and second moments
exp_avg = beta1 * exp_avg + (1 - beta1) * grad
exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2

# Bias correction
step_size = lr * sqrt(1 - beta2^t) / (1 - beta1^t)

# Update rule
param -= step_size * exp_avg / (sqrt(exp_avg_sq) + eps)

# Weight decay
param -= lr * weight_decay * param
```

### AdamWScale Modification

After computing `step_size`, apply RMS scaling (`copied_utils.py:359`):

```python
step_size = step_size * max(1e-3, rms(param))
```

where `rms(tensor) = ||tensor||_2 / sqrt(num_elements)` (`copied_utils.py:305-306`).

### Why This Works

**Problem**: AdamW can diverge during T5 pre-training (observed in prior work and experiments here).

**Root cause**: T5 has parameters of vastly different scales (embeddings, attention weights, feedforward weights). Fixed learning rate doesn't account for this.

**Adafactor's solution**: Scale learning rate by `rms(param)` to adapt to parameter magnitude. This is matrix-wise scaling in Adafactor.

**AdamWScale's approach**: Borrow the RMS scaling from Adafactor but keep AdamW's full second moment (instead of low-rank approximation). This combines:
- AdamW's better second moment estimation (full matrix, not factored)
- Adafactor's adaptive scaling (parameter-magnitude-aware LR)

**Result**: More stable training, faster convergence, better final loss compared to both AdamW alone and Adafactor.

### Trade-offs

- **Memory**: Same as AdamW (2× parameter count for first and second moments)
- **Computation**: Negligible overhead (one extra norm computation per parameter)
- **Hyperparameters**: Same as AdamW (lr, betas, eps, weight_decay)

## Span-Corruption Pre-Training Objective

T5 pre-training uses span-corruption (also called "masked language modeling with spans"). Implementation: `DataCollatorForT5MLM` (`copied_utils.py:15-192`).

### Algorithm

Given a sequence of tokens, randomly mask spans and train the model to predict the masked spans:

1. **Input**: "The quick brown fox jumps over the lazy dog"
2. **Random span masking** (15% of tokens, mean span length 3):
   - Mask spans: [quick brown fox], [lazy]
   - Result: "The <X> jumps over the <Y> dog"
3. **Target**: "<X> quick brown fox <Y> lazy <Z>"
   - Sentinel tokens (<X>, <Y>, <Z>) mark span boundaries
   - EOS token (<Z>) marks end of target

### Implementation Details

**Noise mask generation** (`random_spans_noise_mask()`, `copied_utils.py:127-191`):

```python
num_noise_tokens = round(length * noise_density)  # 15% of tokens
num_noise_spans = round(num_noise_tokens / mean_noise_span_length)  # ~5 spans for length=100

# Partition into noise and non-noise spans
noise_span_lengths = random_partition(num_noise_tokens, num_noise_spans)
nonnoise_span_lengths = random_partition(num_nonnoise_tokens, num_noise_spans)

# Interleave: [nonnoise, noise, nonnoise, noise, ...]
is_noise = compute_interleaved_mask(noise_span_lengths, nonnoise_span_lengths)
```

**Sentinel token assignment** (`create_sentinel_ids()`, `copied_utils.py:88-105`):

Sentinel tokens are `<extra_id_99>`, `<extra_id_98>`, ..., `<extra_id_0>` (100 sentinel tokens in T5 tokenizer). They are assigned in decreasing order to masked spans.

**Input and target lengths** (`compute_input_and_target_lengths()`, `copied_utils.py:194-246`):

To avoid padding, compute exact lengths needed:
- Input length: `num_nonnoise_tokens + num_noise_spans + 1` (EOS)
- Target length: `num_noise_tokens + num_noise_spans + 1` (EOS)

For `input_length=512`, `noise_density=0.15`, `mean_span_length=3.0`:
- Before masking: ~682 tokens
- After masking: 512 tokens (input), ~167 tokens (target)

### Why Span-Corruption

**Advantages over token-level masking (BERT)**:
- More efficient: Predicting spans requires fewer decoder steps
- Harder task: Model must predict contiguous spans, not isolated tokens
- Better for generation: Trains decoder to produce coherent multi-token outputs

**T5 paper result**: Span-corruption outperforms other pre-training objectives (prefix LM, causal LM, BERT-style masking) on downstream tasks.

## Data Collators

### DataCollatorForT5MLM

Pre-training collator (`copied_utils.py:15-192`):

**Input**: Batch of `{"input_ids": [...]}` (tokenized text)

**Output**:
```python
{
    "input_ids": Tensor[batch, input_length],      # Masked input with sentinels
    "labels": Tensor[batch, target_length]         # Masked spans with sentinels
}
```

**Processing**:
1. For each example, generate random span mask
2. Create sentinel IDs for input (marks masked positions)
3. Create sentinel IDs for labels (marks targets)
4. Filter input_ids to remove masked tokens, insert sentinels, add EOS
5. Filter input_ids to keep only masked tokens, insert sentinels, add EOS

**Validation**: Asserts that output shapes match expected `input_length` and `target_length`. If not, raises `ValueError`.

### DataCollatorForNI

Fine-tuning collator for Super-Natural Instructions (`copied_utils.py:400-628`):

**Input**: Batch of task instances with structure:
```python
{
    "Task": task_name,
    "Definition": ["Task description..."],
    "Positive Examples": [{"input": "...", "output": "...", "explanation": "..."}],
    "Negative Examples": [...],
    "Instance": {"input": "...", "output": ["..."]}
}
```

**Output**:
```python
{
    "input_ids": Tensor[batch, max_seq_len],
    "attention_mask": Tensor[batch, max_seq_len],
    "labels": Tensor[batch, max_target_len]  # Masked with -100 for padding
}
```

**Processing**:
1. Format task definition, positive examples, negative examples, and input into a prompt
2. Tokenize and truncate to `max_seq_len`
3. Randomly select one output from multiple references (if available)
4. Tokenize output and mask padding with -100

**Prompt format** (when `add_task_definition=True`, `num_pos_examples=2`):
```
Definition: [Task description from Definition field]

Positive Example 1 -
Input: [Positive example 1 input]
Output: [Positive example 1 output]

Positive Example 2 -
Input: [Positive example 2 input]
Output: [Positive example 2 output]

Now complete the following example -
Input: [Instance input]
Output:
```

**Tk-Instruct mode** (`tk_instruct=True`): Randomly samples from 5 encoding schemas (definition only, examples only, definition+examples, etc.) for each instance to match Tk-Instruct training.

## Learning Rate Schedules

### Cosine Schedule (Recommended)

Implementation: `get_lr_scheduler()` with `lr_scheduler=cosine` (`model_utils.py:306-327`):

```python
# Phase 1: Linear warmup from 0.5*lr to lr
scheduler1 = LinearLR(optimizer, start_factor=0.5, end_factor=1.0, total_iters=warmup_steps)

# Phase 2: Cosine annealing from lr to final_cosine
scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=final_cosine)

# Combine
lr_scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])
```

**Schedule**:
- Steps 0 to `warmup_steps`: Linear increase from `0.5 * base_lr` to `base_lr`
- Steps `warmup_steps` to `total_steps`: Cosine decay from `base_lr` to `final_cosine` (default 1e-5)

**Why cosine**: Smooth decay, no hyperparameter tuning (unlike inverse-sqrt), widely used in modern pre-training.

### Inverse-Square-Root Schedule (T5 Original)

Implementation: `get_lr_scheduler()` with `lr_scheduler=legacy` (`model_utils.py:328-362`):

```python
# Phase 1: Inverse-sqrt with floor at 1e-2
scheduler1 = LambdaLR(optimizer, lambda step: min(1e-2, 1.0 / sqrt(step)) / base_lr if step else 1e-2 / base_lr)

# Phase 2: Linear decay to 0
scheduler2 = LinearLR(optimizer, start_factor=..., end_factor=0, total_iters=...)

# Switch at 90% of training
lr_scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[0.9 * total_steps])
```

**Schedule**:
- Steps 0 to `0.9 * total_steps`: `lr = min(0.01, 1.0 / sqrt(step))`
- Steps `0.9 * total_steps` to `total_steps`: Linear decay to 0

**Why inverse-sqrt**: T5 original schedule. Independent of `base_lr` (logger warns about this).

**Downside**: Requires tuning the 0.9 split point, not as smooth as cosine.

### Constant Schedule (Fine-Tuning)

Implementation: Uses HuggingFace's `get_scheduler()` with `name=constant` (`model_utils.py:363-369`).

**Schedule**: LR remains constant at `base_lr` for all steps.

**Use case**: Fine-tuning on small datasets where LR decay is not needed.

## Training Loop Mechanics

### Gradient Accumulation

Manual gradient accumulation in `train()` (`train_utils.py:175-222`):

```python
optimizer.zero_grad()

for batch_id, batch in enumerate(train_dataloader, start=1):
    loss, stats = forward(model, batch)
    accelerator.backward(loss / grad_acc)  # Scale loss
    train_averager.update(stats)

    if batch_id % grad_acc == 0:
        # Gradient accumulation complete, perform optimizer step
        grad_stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
        train_averager.update(grad_stats)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        maybe_logging(...)
        maybe_eval_predict(...)
        maybe_save_checkpoint(...)

        current_train_step += 1
```

**Key detail**: Loss is scaled by `1 / grad_acc` before backward pass. This ensures gradients have correct magnitude when accumulated.

**Step counting**: `current_train_step` increments only after optimizer step, not after each batch. Total steps = `total_batches / grad_acc`.

### Mixed Precision

Handled automatically by HuggingFace Accelerate via `Accelerator(mixed_precision=args.precision)` (`main.py:25-28`).

**BF16**: Uses `torch.autocast(dtype=torch.bfloat16)` for forward/backward. Gradients and optimizer states remain in FP32.

**TF32**: Automatically enabled for matmul operations on Ampere GPUs via `torch.backends.cuda.matmul.allow_tf32 = True` (`gen_utils.py:28`).

**FP32**: No autocasting, all operations in FP32.

### TF32 Optimization

Enabled in `opti_flags()` (`gen_utils.py:26-29`):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**What is TF32**: Tensor Float 32 is a numeric format on Ampere GPUs (A100, etc.) that uses FP32 range but FP16 precision for matmul. It's a hardware optimization with no code changes needed.

**Speedup**: ~2x over FP32 with negligible loss in accuracy.

### BF16 Flag for Embeddings

If using BF16 and local T5 model, the config is updated with `is_bf16=True` (`gen_utils.py:31-36`). This is used in `T5Stack.forward()` to cast embeddings to BF16 (`t5_model.py:372-373`):

```python
if hasattr(config, "is_bf16") and config.is_bf16:
    inputs_embeds = inputs_embeds.to(torch.bfloat16)
```

**Why**: Accelerate's autocast doesn't apply to embeddings. Explicit cast ensures embeddings match the precision of the rest of the model.

## Evaluation Metrics

### Pre-Training: Negative Log-Likelihood

Computed in `eval()` (`train_utils.py:107-123`):

```python
for batch in test_dataloader:
    outputs = model(**batch)
    loss = outputs.loss  # CrossEntropyLoss on decoder outputs
    stats["loss"] = loss.item()
```

**Metric**: Average loss on held-out C4 validation set. Lower is better.

**Comparison**: T5 v1.1 paper reports 1.942 NLL. This repo achieves 1.953 (AdamWScale + cosine) or 1.995 (Adafactor + ISR).

### Fine-Tuning: ROUGE-L

Computed in `predict()` (`train_utils.py:126-172`):

```python
for batch in test_dataloader:
    predictions = model.generate(input_ids, attention_mask, max_length=max_target_len)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

    metric.add_batch(predictions=predictions, references=references)

rougeL = metric.compute(use_stemmer=True, use_aggregator=False)["rougeL"]
```

**Metric**: ROUGE-L (longest common subsequence F1-score) averaged over test set. Higher is better.

**Comparison**: T5 v1.1 base achieves 40.9 ROUGE-L on SNI. This repo achieves 40.7 after 16h pre-training.

## Performance Optimizations

### Streaming Data Pipeline

C4 dataset is 300+ GB. Downloading and preprocessing before training would take hours.

**Solution**: `datasets.load_dataset("c4", "en", streaming=True)` returns an `IterableDataset` that streams data on-the-fly (`model_utils.py:106-123`).

**Benefits**:
- Training starts immediately (no download wait)
- Preprocessing (tokenization, masking) happens in parallel with training
- Disk usage: minimal (only buffers a few batches)

**Trade-offs**:
- No random access (must use shuffling with finite buffer)
- No `len(dataset)` (must train for fixed number of steps, not epochs)

### PyTorch 2.0 Compile

Enabled via `model.compile=true` (`main.py:52-53`):

```python
if args.model.compile:
    model = torch.compile(model)
```

**What it does**: `torch.compile` traces the model and generates optimized CUDA kernels using TorchInductor.

**Speedup**: ~2x faster per-step time (0.56s vs 1.30s for BF16 training).

**Caveats**:
- First few steps are slow (compilation overhead)
- Can cause issues with dynamic shapes or control flow
- Not compatible with some operations (e.g., certain RNN variants)

### Data Loader Workers

DataLoader uses `num_workers=8` (`model_utils.py:234`) to parallelize data loading:

```python
DataLoader(dataset, num_workers=8, pin_memory=True, ...)
```

**Effect**: Data preprocessing (tokenization, masking) happens in parallel across 8 worker processes, overlapping with GPU computation.

**Requirement**: Sufficient CPU cores (8+ cores recommended).

### Gradient Checkpointing

**Not implemented**. Gradient checkpointing trades compute for memory by recomputing activations during backward pass. This would reduce memory usage but is not needed for single-GPU training of T5-base.

Could be added for larger models by enabling `use_cache=False` and wrapping encoder/decoder blocks with `torch.utils.checkpoint.checkpoint()`.
