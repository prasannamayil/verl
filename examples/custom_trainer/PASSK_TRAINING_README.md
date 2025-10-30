# Pass@k Analytical Advantage Training

## Overview

This implementation adds a new advantage estimator for Pass@k training based on analytical derivation. The algorithm uses bootstrap sampling with analytical formulas to compute advantages for policy gradient training, specifically designed for optimizing Pass@k metrics.

## Algorithm Description

The Pass@k analytical advantage estimator implements Algorithm 3 from the paper, which computes advantages using:

1. **Group Statistics** (Equations 11-12):
   - Average reward of groups: \( \bar{R}^{group} = 1 - \frac{\binom{N_{neg}}{k}}{\binom{N_{rollout}}{k}} \)
   - Standard deviation: \( \sigma^{group} = \sqrt{\bar{R}^{group} \times (1 - \bar{R}^{group})} \)

2. **Response-Relative Advantages** (Equations 14-15):
   - Positive response advantage: \( \hat{A}_{pos} = \frac{1 - \bar{R}^{group}}{\sigma^{group}} \)
   - Negative response advantage: \( \hat{A}_{neg} = \frac{-\bar{R}^{group}}{\sigma^{group}} \)

## Implementation Details

### Files Modified

1. **`verl/trainer/ppo/core_algos.py`**:
   - Added `PASSK_ANALYTICAL` to `AdvantageEstimator` enum
   - Implemented `compute_passk_analytical_advantage()` function (lines 421-521)
   - Uses scipy.special.comb for binomial coefficient calculations
   - Handles edge cases (insufficient samples, all same class)

2. **`verl/trainer/config/ppo_trainer.yaml`**:
   - Added `passk_k`: k value for Pass@k metric (default: 4)
   - Added `passk_reward_threshold`: threshold to classify positive/negative responses (default: 0.5)

3. **`examples/custom_trainer/run_passk_qwen25math_1.5b.sh`**:
   - Complete training script for Pass@k training
   - Configured for Qwen2.5-Math-1.5B model
   - Uses 8 rollouts with Pass@4 metric

## Usage

### Basic Usage

```bash
cd /fast/pmayilvahanan/verl
./examples/custom_trainer/run_passk_qwen25math_1.5b.sh
```

### Key Configuration Parameters

```yaml
algorithm:
  adv_estimator: passk_analytical
  passk_k: 4                      # k for Pass@k (e.g., Pass@4, Pass@8)
  passk_reward_threshold: 0.5     # Threshold to classify responses as positive/negative
```

### Customization Options

#### Change k value for Pass@k:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=passk_analytical \
    algorithm.passk_k=8 \
    ...
```

#### Adjust reward threshold:
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=passk_analytical \
    algorithm.passk_reward_threshold=0.7 \
    ...
```

#### Use with GSPO loss (sequence-level):
```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=passk_analytical \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    ...
```

## Requirements

- **Minimum rollouts**: Must have at least k rollouts per prompt (k = passk_k)
- **scipy**: Required for binomial coefficient calculations (`scipy.special.comb`)
- **Reward format**: Binary or continuous rewards that can be thresholded

## Algorithm Flow

1. **Rollout Phase**: Generate n responses per prompt (n ≥ k)
2. **Reward Computation**: Compute rewards for all responses
3. **Classification**: Classify responses as positive (reward > threshold) or negative
4. **Group Statistics**: Calculate average reward and std dev of virtual groups
5. **Advantage Assignment**: 
   - Positive responses get advantage: (1 - R̄_group) / σ_group
   - Negative responses get advantage: -R̄_group / σ_group
6. **Policy Update**: Standard PPO update with computed advantages

## Differences from GRPO_PASSK

The existing `GRPO_PASSK` estimator:
- Only assigns non-zero advantage to the best response per group
- Uses r_max - r_second_max as advantage
- Based on https://arxiv.org/abs/2503.19595

The new `PASSK_ANALYTICAL` estimator:
- Assigns advantages to ALL responses (positive and negative)
- Uses analytical formulas based on group statistics
- Removes variance from bootstrap sampling operation
- Based on Algorithm 3 with analytical derivation

## Performance Considerations

- **Memory**: O(batch_size) for storing scores and indices per group
- **Computation**: O(batch_size) with small constant for binomial coefficients
- **Numerical Stability**: Uses epsilon=1e-6 for division safety

## Troubleshooting

### Error: "Pass@k requires at least k samples per group"
**Solution**: Increase `actor_rollout_ref.rollout.n` to be ≥ passk_k

### Error: "All responses are same class"
**Solution**: 
- Adjust `algorithm.passk_reward_threshold`
- Check reward function is producing varied outputs

### Warning: "All responses positive/negative"
**Solution**: The algorithm will skip computing advantages for such groups (assigns zero)

## Example Configurations

### Pass@4 with Binary Rewards:
```yaml
algorithm.adv_estimator: passk_analytical
algorithm.passk_k: 4
algorithm.passk_reward_threshold: 0.5
actor_rollout_ref.rollout.n: 8  # Generate 8 samples
```

### Pass@8 with Stricter Threshold:
```yaml
algorithm.adv_estimator: passk_analytical
algorithm.passk_k: 8
algorithm.passk_reward_threshold: 0.8
actor_rollout_ref.rollout.n: 16  # Generate 16 samples
```

### Pass@k with GSPO Loss:
```yaml
algorithm.adv_estimator: passk_analytical
algorithm.passk_k: 4
actor_rollout_ref.actor.policy_loss.loss_mode: gspo
actor_rollout_ref.actor.loss_agg_mode: seq-mean-token-mean
```

## References

- Algorithm 3: The Pseudo Code for Pass@k Training with Analytical Derivation
- Section 2.4: Analytical Derivation of Efficient and Effective Pass@k Training
- Equations 11, 12, 14, 15: Group statistics and response advantages

## Contact & Support

For issues or questions about this implementation, please check:
1. Logs in `checkpoint_dir/validation_data/` for validation metrics
2. wandb dashboard for training curves
3. Console output for any error messages

