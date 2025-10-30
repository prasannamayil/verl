# Entropy-Based Advantage Shaping for VERL

## Quick Start

Run the entropy-based advantage shaping method with:

```bash
cd /fast/pmayilvahanan/verl
bash examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh
```

## What Was Implemented

This implements the entropy-based advantage shaping method from the provided pseudo code:

```
adv += min(alpha * entropy.detach(), |adv|/kappa)
```

The method modifies advantages before policy loss computation to encourage exploration by adding an entropy-based bonus term.

## Key Configuration Parameters

### Enable the Method

```bash
actor_rollout_ref.actor.use_entropy_advantage_shaping=true
```

### Set Hyperparameters

```bash
actor_rollout_ref.actor.entropy_advantage_alpha=0.1   # Alpha coefficient
actor_rollout_ref.actor.entropy_advantage_kappa=1.0   # Kappa coefficient
```

### Compatibility

Works with any advantage estimator:
```bash
algorithm.adv_estimator=gae           # Generalized Advantage Estimation
algorithm.adv_estimator=grpo          # Group Relative Policy Optimization
algorithm.adv_estimator=reinforce_plus_plus  # REINFORCE++
# ... and others
```

## Example: Running with Different Configurations

### Basic PPO + Entropy Shaping

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.actor.use_entropy_advantage_shaping=true \
    actor_rollout_ref.actor.entropy_advantage_alpha=0.1 \
    actor_rollout_ref.actor.entropy_advantage_kappa=1.0 \
    # ... other standard PPO configs
```

### GRPO + Entropy Shaping

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.use_entropy_advantage_shaping=true \
    actor_rollout_ref.actor.entropy_advantage_alpha=0.15 \
    actor_rollout_ref.actor.entropy_advantage_kappa=0.8 \
    # ... other GRPO configs
```

### With KL Penalty

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_ctrl.kl_coef=0.01 \
    actor_rollout_ref.actor.use_entropy_advantage_shaping=true \
    actor_rollout_ref.actor.entropy_advantage_alpha=0.1 \
    actor_rollout_ref.actor.entropy_advantage_kappa=1.0 \
    # ... other configs
```

## Hyperparameter Tuning Guide

### Alpha (`entropy_advantage_alpha`)

- **Default**: 0.1
- **Range**: 0.01 - 0.5
- **Effect**: Controls the magnitude of entropy bonus
  - Lower (0.01-0.05): Subtle exploration boost
  - Medium (0.1-0.2): Balanced exploration
  - Higher (0.3-0.5): Strong exploration bias

**When to increase**: Tasks requiring more exploration (creative generation, diverse solutions)  
**When to decrease**: Tasks with clear optimal solutions (accuracy-focused tasks)

### Kappa (`entropy_advantage_kappa`)

- **Default**: 1.0
- **Range**: 0.5 - 2.0
- **Effect**: Controls relative scaling of entropy vs advantage magnitude
  - Lower (0.5-0.8): Entropy term has more influence
  - Medium (1.0): Equal scaling
  - Higher (1.5-2.0): Advantage magnitude dominates

**When to increase**: When you want entropy to have less impact on already strong advantages  
**When to decrease**: When you want consistent exploration regardless of advantage magnitude

## Implementation Details

### Files Modified

1. **Configuration**:
   - `verl/workers/config/actor.py`: Added 3 config parameters
   - `verl/trainer/config/actor/actor.yaml`: Added YAML configuration

2. **Actor Implementation**:
   - `verl/workers/actor/dp_actor.py`: FSDP/DataParallel backend
   - `verl/workers/actor/megatron_actor.py`: Megatron backend

3. **Example Script**:
   - `examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh`: Reference script

### How It Works

1. **Entropy Computation**: When `use_entropy_advantage_shaping=true`, entropy is computed for all tokens
2. **Advantage Shaping**: Before computing policy loss, advantages are modified:
   ```python
   entropy_term = min(alpha * entropy.detach(), |advantages| / kappa)
   shaped_advantages = advantages + entropy_term
   ```
3. **Policy Update**: The shaped advantages are used for policy gradient computation

### Key Design Decisions

- **Detached entropy**: Uses `entropy.detach()` to prevent gradient flow through the entropy term
- **Additive shaping**: Always adds to advantages (never subtracts)
- **Bounded term**: The `min()` operation prevents the entropy term from dominating
- **Token-level**: Applied independently to each token in the response

## Validation

All modified files pass syntax validation:

```
✓ verl/workers/config/actor.py syntax is valid
✓ verl/workers/actor/dp_actor.py syntax is valid
✓ verl/workers/actor/megatron_actor.py syntax is valid
```

No linter errors detected.

## Expected Results

### Training Behavior

- **Exploration**: Model will explore more during training
- **Entropy**: Higher entropy in generated tokens (especially early in training)
- **Convergence**: May take slightly longer to converge (due to exploration)
- **Final Performance**: Better performance on tasks requiring diverse solutions

### Monitoring

Key metrics to watch:
- `actor/entropy`: Should be higher than baseline
- `actor/pg_loss`: May be slightly higher initially
- Validation metrics: Should improve with proper tuning

## Troubleshooting

### Issue: Training is unstable

**Solution**: Reduce `alpha` (try 0.05 or 0.01)

### Issue: Not enough exploration

**Solution**: Increase `alpha` (try 0.2 or 0.3) or decrease `kappa` (try 0.5)

### Issue: Too much exploration, not converging

**Solution**: Decrease `alpha` (try 0.05) or increase `kappa` (try 1.5 or 2.0)

### Issue: Memory issues

**Note**: Entropy computation requires additional memory. If you encounter OOM:
- Reduce `ppo_micro_batch_size_per_gpu`
- Enable `entropy_checkpointing=true`
- Consider using smaller models or reducing sequence length

## Comparison with Baseline

To compare with baseline (no entropy shaping):

```bash
# Baseline
bash examples/custom_trainer/run_gspo_qwen2_7b.sh

# With entropy shaping
bash examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh
```

Compare final validation metrics to assess effectiveness.

## Citation

If you use entropy-based advantage shaping in your research, please cite the original paper and this implementation.

## Support

For issues or questions about the implementation:
1. Check syntax validation passed
2. Verify configuration in logs
3. Try default hyperparameters first
4. Refer to `/fast/pmayilvahanan/verl/ENTROPY_SHAPING_IMPLEMENTATION.md` for detailed technical documentation


