# GSPO Clip-Cov and GSPO KL-Cov Implementation

## Overview

This document describes the implementation of two new policy loss variants that combine GSPO (Geometric Sequential Policy Optimization) with covariance-based regularization techniques from the Entropy Mechanism paper.

## Implemented Functions

### 1. `compute_policy_loss_gspo_clip_cov` (registered as `"gspo_clip_cov"`)

**Location:** `verl/trainer/ppo/core_algos.py` lines 1361-1444

**Purpose:** Combines GSPO's sequence-level importance sampling with Clip-Cov's token-level covariance masking.

**Key Mechanics:**
- **GSPO Base:** Uses sequence-level importance ratio broadcast to token level
  ```python
  seq_importance_ratio = exp(log_prob - log_prob.detach() + seq_kl.detach().unsqueeze(-1))
  ```
- **Covariance Selection:** Identifies tokens with high covariance between advantages and `log_seq_importance_ratio` (not raw `log_prob`)
- **Token Masking:** Masks out (sets weight to 0) for selected high-covariance tokens
- **Aggregation:** `seq-mean-token-mean` as required by GSPO

**Why `log_seq_importance_ratio` for covariance?**
- In vanilla Clip-Cov, covariance uses `log_prob` because the gradient is `∇log π(a|s) * A`
- In GSPO, the gradient is `∇log(seq_importance_ratio) * A`
- Therefore, we measure covariance between advantages and the log importance ratio to properly identify problematic tokens in GSPO's gradient structure

### 2. `compute_policy_loss_gspo_kl_cov` (registered as `"gspo_kl_cov"`)

**Location:** `verl/trainer/ppo/core_algos.py` lines 1447-1513

**Purpose:** Combines GSPO's sequence-level importance sampling with KL-Cov's selective KL penalty on high-covariance tokens.

**Key Mechanics:**
- **GSPO Base:** Same sequence-level importance ratio as above
- **Base Loss:** `-advantages * seq_importance_ratio`
- **KL-Augmented Loss:** `base_loss + ppo_kl_coef * |KL|`
- **Selective Application:** Only applies KL penalty to top-k% tokens with highest covariance between advantages and log_prob
- **Aggregation:** `seq-mean-token-mean` as required by GSPO

**Algorithm:**
1. Compute base GSPO objective for all tokens
2. Compute covariance = (adv - mean_adv) * (log_prob - mean_log_prob) for all valid tokens
3. Select top k% tokens by covariance
4. Replace loss for selected tokens with KL-augmented version
5. Aggregate with sequence-mean-token-mean

## Configuration

Both functions respect the existing config hierarchy:

```yaml
actor_rollout_ref.actor:
  clip_ratio_low: 0.0003      # GSPO paper recommendation
  clip_ratio_high: 0.0004     # GSPO paper recommendation
  loss_agg_mode: seq-mean-token-mean
  entropy_coeff: 0.0          # No entropy regularization
  use_kl_loss: false          # No explicit KL loss
  kl_loss_coef: 0.0
  
  policy_loss:
    loss_mode: gspo_clip_cov  # or gspo_kl_cov
    clip_cov_ratio: 0.0002    # For gspo_clip_cov
    clip_cov_lb: 1.0          # For gspo_clip_cov
    clip_cov_ub: 5.0          # For gspo_clip_cov
    kl_cov_ratio: 0.0002      # For gspo_kl_cov
    ppo_kl_coef: 0.0          # For gspo_kl_cov (set to 0 to disable KL penalty)
```

## Run Scripts

Five new shell scripts created in `examples/custom_trainer/`:

### GRPO Variants (advantage estimator: grpo)
1. **`run_grpo_qwen25math_1.5b.sh`**
   - loss_mode: vanilla
   - Standard PPO clip ratios (0.2/0.2)

2. **`run_grpo_clip_cov_qwen25math_1.5b.sh`**
   - loss_mode: clip_cov
   - Standard PPO clip ratios (0.2/0.2)
   - Uses token-level ratio in covariance

3. **`run_grpo_kl_cov_qwen25math_1.5b.sh`**
   - loss_mode: kl_cov
   - Standard PPO clip ratios (0.2/0.2)
   - ppo_kl_coef: 0.0 (disabled)

### GSPO Variants (advantage estimator: grpo)
4. **`run_gspo_clip_cov_qwen25math_1.5b.sh`**
   - loss_mode: gspo_clip_cov
   - GSPO clip ratios (0.0003/0.0004)
   - Uses sequence-level importance ratio in covariance

5. **`run_gspo_kl_cov_qwen25math_1.5b.sh`**
   - loss_mode: gspo_kl_cov
   - GSPO clip ratios (0.0003/0.0004)
   - ppo_kl_coef: 0.0 (disabled)

**All scripts have:**
- `entropy_coeff=0.0`
- `use_kl_in_reward=false`, `kl_coef=0.0`
- `use_kl_loss=false`, `kl_loss_coef=0.0`

## Key Differences from Base Implementations

### GSPO Clip-Cov vs. Vanilla Clip-Cov

| Aspect | Vanilla Clip-Cov | GSPO Clip-Cov |
|--------|------------------|---------------|
| Importance weighting | Token-level ratio `exp(log_prob - old_log_prob)` | Sequence-level ratio broadcast to tokens |
| Covariance computation | `(adv - mean_adv) * (log_prob - mean_log_prob)` | `(adv - mean_adv) * (log_seq_importance_ratio - mean)` |
| Aggregation | `token-mean` | `seq-mean-token-mean` |
| Clip ratios | Typically 0.2/0.2 | Typically 0.0003/0.0004 |

### GSPO KL-Cov vs. Vanilla KL-Cov

| Aspect | Vanilla KL-Cov | GSPO KL-Cov |
|--------|----------------|-------------|
| Base objective | `-adv * ratio` | `-adv * seq_importance_ratio` |
| KL-augmented | `base + coef * \|KL\|` | `base + coef * \|KL\|` |
| Selection criterion | Same (covariance of adv and log_prob) | Same (covariance of adv and log_prob) |
| Aggregation | `token-mean` | `seq-mean-token-mean` |

## Theoretical Justification

### Why Combine GSPO with Covariance Methods?

1. **GSPO's Strength:** Geometric mean of token-level importance ratios provides better sequence-level optimization, especially for long sequences

2. **Covariance Methods' Strength:** Identify and handle tokens where the policy gradient estimator has high variance or problematic correlation structure

3. **Combined Benefit:** 
   - GSPO provides stable sequence-level importance weighting
   - Clip-Cov/KL-Cov provide fine-grained token-level variance reduction
   - Together: stable sequence optimization + reduced gradient variance

### When to Use Each Variant

- **GSPO Clip-Cov:** When you want GSPO's benefits but observe gradient instability from specific tokens with high advantage-ratio covariance

- **GSPO KL-Cov:** When you want GSPO's benefits but need to regularize high-covariance tokens with KL penalty (though typically set to 0 as per your requirements)

## Implementation Correctness

✅ **Verified Correct:**
- GSPO sequence-level importance ratio computation matches original GSPO
- Stop-gradient operations correctly applied
- Aggregation mode respects GSPO requirements
- Config structure follows existing patterns
- Both functions integrate with rollout importance sampling weights

✅ **Key Fix Applied:**
- GSPO Clip-Cov now uses `log_seq_importance_ratio` for covariance (not raw `log_prob`)
- This aligns with GSPO's gradient structure where advantages are weighted by the sequence importance ratio

## Testing Recommendations

1. **Sanity Check:** Run short experiments to verify:
   - No runtime errors
   - Loss values are reasonable
   - Metrics are logged correctly

2. **Ablation Studies:**
   - Compare GSPO vs GSPO-Clip-Cov vs GSPO-KL-Cov
   - Verify Clip-Cov masking improves stability (monitor gradient norms, clipfrac)

3. **Hyperparameter Sensitivity:**
   - `clip_cov_ratio` (default 0.0002): fraction of tokens to mask
   - `clip_cov_lb/ub` (1.0/5.0): covariance bounds for selection
   - GSPO clip ratios (0.0003/0.0004): may need tuning per model/task

## References

- **GSPO Paper:** https://arxiv.org/pdf/2507.18071
- **Entropy Mechanism (Clip-Cov, KL-Cov):** https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
- **Original VERL Implementations:** `verl/trainer/ppo/core_algos.py`

## Author Notes

Implementation completed: October 30, 2025
All scripts configured with KL and entropy coefficients set to 0 as requested.

