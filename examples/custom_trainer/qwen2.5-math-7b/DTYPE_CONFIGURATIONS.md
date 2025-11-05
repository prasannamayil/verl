# Dtype Configuration Guide for VERL Training

This guide explains how to configure data types (dtypes) for different components in VERL training pipelines, particularly for GSPO and other PPO-based algorithms.

## Quick Reference

### Four Key Dtype Settings

1. **`actor.dtype`**: Computation dtype for actor training (autocast, gradient scaler)
2. **`actor.fsdp_config.model_dtype`**: Storage dtype for actor model parameters (FSDP mixed precision)
3. **`rollout.dtype`**: Dtype for generation/rollout (vLLM inference engine)
4. **`ref.fsdp_config.model_dtype`**: Storage dtype for reference model parameters

## Common Configurations

### Configuration 1: FP16 Everywhere (Default, Best Memory/Speed)

**Use case:** Maximum memory efficiency, fastest training, suitable for most cases with gradient scaling

```bash
actor_rollout_ref.actor.dtype=float16 \
actor_rollout_ref.actor.fsdp_config.model_dtype=fp16 \
actor_rollout_ref.rollout.dtype=float16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=fp16
```

**Characteristics:**
- ✅ ~50% memory reduction vs FP32
- ✅ Fastest training and inference
- ✅ Automatic gradient scaling for numerical stability
- ⚠️ May have numerical issues with very long sequences (>4096 tokens) without proper tuning
- **Recommended for:** Standard GSPO, GRPO training with 1024-3072 token sequences

---

### Configuration 2: Old Default (FP32 Training, BF16 Inference)

**Use case:** Maximum training stability, good inference efficiency

```bash
actor_rollout_ref.actor.dtype=float32 \
actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

**Characteristics:**
- ✅ Maximum numerical stability for training
- ✅ No gradient scaling needed
- ❌ Higher memory usage (~2x vs FP16 for actor)
- ✅ Efficient inference with BF16
- **Recommended for:** Debugging, very long sequences, numerical stability issues

---

### Configuration 3: BF16 Everywhere (Good Balance)

**Use case:** Balanced stability and efficiency

```bash
actor_rollout_ref.actor.dtype=bfloat16 \
actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

**Characteristics:**
- ✅ Better numerical stability than FP16 (same dynamic range as FP32)
- ✅ ~50% memory reduction vs FP32
- ✅ No gradient scaling needed
- ⚠️ Slightly slower than FP16 on some hardware
- **Recommended for:** Long sequences, GSPO with >3072 tokens, stability-critical applications

---

### Configuration 4: Mixed Precision (FP32 Training, FP16 Inference)

**Use case:** Stable training with fast/efficient inference

```bash
actor_rollout_ref.actor.dtype=float32 \
actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
actor_rollout_ref.rollout.dtype=float16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
```

**Characteristics:**
- ✅ Stable FP32 training
- ✅ Fast FP16 generation
- ❌ Higher actor memory usage
- ✅ Good for memory-constrained generation
- **Recommended for:** Large batch sizes for generation, small batch for training

---

## Understanding Each Setting

### 1. `actor.dtype` (Computation Dtype)

**Controls:** The dtype used during forward and backward passes via PyTorch's autocast

**Values:** `float16`, `bfloat16`, `float32`

**Impact:**
- `float16`: Enables `ShardedGradScaler` for automatic gradient scaling (growth_interval=400)
- `bfloat16`: No gradient scaling, uses BF16 autocast
- `float32`: No autocast, full precision computation

**Where it's used:**
- Forward pass computations
- Gradient computations
- Gradient scaling decisions

---

### 2. `actor.fsdp_config.model_dtype` (Parameter Storage)

**Controls:** How model parameters are stored in FSDP's mixed precision policy

**Values:** `fp32`, `fp16`, `bf16` (shorthand notation)

**Impact:**
- Determines memory footprint of model parameters
- Typically should match `actor.dtype` for efficiency
- Can differ for specific memory management strategies

**Where it's used:**
- FSDP MixedPrecision policy (`param_dtype`)
- Parameter sharding and gathering
- Checkpoint storage

---

### 3. `rollout.dtype` (Generation Dtype)

**Controls:** Dtype for vLLM/SGLang inference engine during generation

**Values:** `float16`, `bfloat16`, `float32` (PyTorch names, NOT shorthand)

**Impact:**
- KV cache memory usage
- Generation speed
- Inference numerical precision

**Important:** Must use PyTorch dtype names (`float16`), not shorthand (`fp16`)

**Where it's used:**
- vLLM/SGLang model loading
- KV cache allocation
- Generation forward passes

---

### 4. `ref.fsdp_config.model_dtype` (Reference Model Storage)

**Controls:** Storage dtype for reference model (used for KL penalties)

**Values:** `fp32`, `fp16`, `bf16`

**Impact:**
- Reference model memory usage
- Usually set to efficient dtype since it's inference-only

**Typical values:**
- `bf16`: Good balance (recommended)
- `fp16`: Maximum efficiency
- `fp32`: Maximum precision (rarely needed)

---

## FP16 Specific Features

When using `actor.dtype=float16`, the following features are automatically enabled:

### Gradient Scaling

```python
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
self.scaler = ShardedGradScaler(growth_interval=400)
```

**What it does:**
- Scales loss before backward pass to prevent gradient underflow
- Unscales gradients before clipping
- Dynamically adjusts scale factor based on gradient norms
- Growth interval of 400 provides stable scaling for RL workloads

### Modified Training Loop

1. **Before backward:** `scaler.scale(loss).backward()`
2. **Before optimizer step:** `scaler.unscale_(optimizer)` → gradient clipping → `scaler.step(optimizer)` → `scaler.update()`
3. **Non-finite gradient handling:** Automatic skip of optimizer step if gradients are NaN/Inf

---

## Choosing the Right Configuration

### For Standard GSPO/GRPO Training
- **Start with:** Configuration 1 (FP16 everywhere)
- **Sequence length:** < 3072 tokens → FP16 works well
- **Sequence length:** > 3072 tokens → Consider BF16 (Configuration 3)

### For Debugging NaN Issues
1. Try Configuration 2 (FP32 training) to isolate numerical issues
2. Check if NaNs persist → likely algorithmic issue, not dtype
3. If NaNs disappear → consider Configuration 3 (BF16) as middle ground

### For Memory-Constrained Setups
- **Training memory:** Use Configuration 1 or 3
- **Inference memory:** Set `rollout.dtype=float16` regardless of training dtype
- **Critical memory:** Consider offloading: `fsdp_config.param_offload=True`

### For Very Long Context (>4096 tokens)
- **Recommended:** Configuration 3 (BF16 everywhere)
- **Alternative:** Configuration 2 (FP32 training) if BF16 still shows instability

---

## Validation and Testing

All configurations include validation during generation (testing):
- Validation uses the **same rollout dtype** as training
- No separate validation dtype configuration needed
- Metrics computed in full precision regardless of generation dtype

---

## Migration from Old Defaults

### Before FP16 Patch
```bash
# Implicit defaults:
# actor.dtype = (not set, autocast to bfloat16)
# actor.fsdp_config.model_dtype = fp32
# rollout.dtype = bfloat16
# ref.fsdp_config.model_dtype = (inherits from defaults)
```

### After FP16 Patch (Current Defaults)
```bash
# New defaults:
# actor.dtype = float16
# actor.fsdp_config.model_dtype = fp32 (can override to fp16)
# rollout.dtype = float16
# ref.fsdp_config.model_dtype = fp32 (can override to fp16)
```

### To Restore Old Behavior
Use Configuration 2 exactly as shown above.

---

## Troubleshooting

### NaN Gradients with FP16
1. Check sequence length → if >3072, try BF16
2. Verify gradient scaling is enabled (should be automatic)
3. Try increasing gradient clipping: `actor.grad_clip=2.0`
4. Last resort: Switch to Configuration 2 (FP32)

### OOM (Out of Memory)
1. Use Configuration 1 (FP16 everywhere)
2. Enable offloading: `actor.fsdp_config.param_offload=True`
3. Reduce micro batch size: `ppo_micro_batch_size_per_gpu=4`
4. Enable activation offloading: `model.enable_activation_offload=True`

### vLLM Dtype Error
```
ValueError: Unknown dtype: fp16
```
**Solution:** Use PyTorch dtype name: `rollout.dtype=float16` (not `fp16`)

### FSDP Config Error
```
FSDPActorConfig.__init__() got an unexpected keyword argument 'dtype'
```
**Solution:** Ensure you've updated to the patched version with `actor.dtype` field in `ActorConfig` dataclass

---

## References

- [FP16-FP16 Training Paper](https://arxiv.org/abs/2501.10491) - Source of the gradient scaling patch
- GSPO Paper: Recommended clip ratios (0.0003-0.0004) are sensitive to dtype
- PyTorch FSDP MixedPrecision: https://pytorch.org/docs/stable/fsdp.html
- vLLM Quantization: https://docs.vllm.ai/en/latest/

---

## Summary Table

| Configuration | Actor Train | Actor Storage | Rollout | Ref Storage | Use Case |
|---------------|-------------|---------------|---------|-------------|----------|
| **FP16 All** | float16 | fp16 | float16 | fp16 | Best memory/speed |
| **Old Default** | float32 | fp32 | bfloat16 | bf16 | Max stability |
| **BF16 All** | bfloat16 | bf16 | bfloat16 | bf16 | Balanced |
| **Mixed** | float32 | fp32 | float16 | bf16 | Stable train, fast gen |

---

*Last updated: October 2025*
*VERL version: 1.0+ (with FP16 patch)*




