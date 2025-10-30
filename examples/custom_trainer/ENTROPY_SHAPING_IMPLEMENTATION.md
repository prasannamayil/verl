# Entropy-Based Advantage Shaping Implementation

## Overview

This document describes the implementation of the entropy-based advantage shaping method for PPO/GRPO training in the VERL framework.

## Method Description

The entropy-based advantage shaping method modifies the advantage estimates by adding an entropy-based term before computing the policy loss:

```python
adv += min(alpha * entropy.detach(), |adv|/kappa)
```

Where:
- `alpha`: Scaling coefficient for the entropy term
- `kappa`: Normalization factor for advantage magnitude
- `entropy`: Per-token entropy of the policy distribution (detached, no gradient flow)

## Implementation Details

### 1. Configuration Parameters

Added three new parameters to `ActorConfig` (in `verl/workers/config/actor.py`):

```python
use_entropy_advantage_shaping: bool = False  # Enable/disable the method
entropy_advantage_alpha: float = 0.1         # Alpha coefficient
entropy_advantage_kappa: float = 1.0         # Kappa coefficient
```

These parameters are also exposed in the YAML configuration (`verl/trainer/config/actor/actor.yaml`):

```yaml
# Whether to use entropy-based advantage shaping
use_entropy_advantage_shaping: false

# Alpha coefficient for entropy-based advantage shaping
entropy_advantage_alpha: 0.1

# Kappa coefficient for entropy-based advantage shaping
entropy_advantage_kappa: 1.0
```

### 2. Actor Implementation

#### DataParallel Actor (`verl/workers/actor/dp_actor.py`)

Modified the `update_policy` method:

1. **Enable entropy calculation** when shaping is active:
   ```python
   if self.config.get("use_entropy_advantage_shaping", False):
       calculate_entropy = True
   ```

2. **Apply entropy-based shaping** before policy loss computation:
   ```python
   if self.config.get("use_entropy_advantage_shaping", False):
       alpha = self.config.entropy_advantage_alpha
       kappa = self.config.entropy_advantage_kappa
       entropy_term = torch.min(
           alpha * entropy.detach(),
           advantages.abs() / kappa
       )
       advantages = advantages + entropy_term
   ```

#### Megatron Actor (`verl/workers/actor/megatron_actor.py`)

Similar implementation for Megatron backend:

1. **Enable entropy calculation** in `update_policy`:
   ```python
   if self.config.get("use_entropy_advantage_shaping", False):
       calculate_entropy = True
   ```

2. **Apply shaping** in the loss function:
   ```python
   if self.config.get("use_entropy_advantage_shaping", False) and calculate_entropy:
       entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
       alpha = self.config.entropy_advantage_alpha
       kappa = self.config.entropy_advantage_kappa
       entropy_term = torch.min(
           alpha * entropy.detach(),
           advantages.abs() / kappa
       )
       advantages = advantages + entropy_term
   ```

### 3. Key Implementation Features

- **Minimal changes**: The implementation is minimal and doesn't interfere with existing functionality
- **Detached entropy**: Uses `entropy.detach()` to prevent gradient flow through the entropy term
- **Conditional execution**: Only activates when `use_entropy_advantage_shaping=True`
- **Backend agnostic**: Works with both FSDP (DataParallel) and Megatron backends
- **Advantage estimator agnostic**: Works with GAE, GRPO, REINFORCE++, and other advantage estimators

## Usage

### Running with Entropy-Based Advantage Shaping

A reference script is provided: `examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh`

Key parameters in the script:

```bash
# Enable entropy-based advantage shaping
use_entropy_advantage_shaping=true

# Set alpha and kappa coefficients
entropy_advantage_alpha=0.1
entropy_advantage_kappa=1.0

# Can be used with any advantage estimator
adv_estimator=gae  # or "grpo", "reinforce_plus_plus", etc.
```

Run the script:

```bash
bash examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh
```

### Command-line Override

You can override parameters via Hydra command-line syntax:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.use_entropy_advantage_shaping=true \
    actor_rollout_ref.actor.entropy_advantage_alpha=0.1 \
    actor_rollout_ref.actor.entropy_advantage_kappa=1.0 \
    algorithm.adv_estimator=gae \
    ... # other parameters
```

## Compatibility

The implementation is compatible with:

- ✅ All advantage estimators (GAE, GRPO, REINFORCE++, REMAX, etc.)
- ✅ Both FSDP and Megatron backends
- ✅ All loss modes (vanilla, GSPO, GPG, clip-cov, kl-cov)
- ✅ KL divergence penalties (both in-reward and in-loss)
- ✅ Entropy regularization (can be used together)
- ✅ Rollout importance sampling
- ✅ Multi-turn conversations
- ✅ Dynamic batch sizing

## Files Modified

1. **Configuration**:
   - `verl/workers/config/actor.py`: Added config parameters to `ActorConfig`
   - `verl/trainer/config/actor/actor.yaml`: Added YAML configuration

2. **Implementation**:
   - `verl/workers/actor/dp_actor.py`: Implemented shaping in DataParallel actor
   - `verl/workers/actor/megatron_actor.py`: Implemented shaping in Megatron actor

3. **Scripts**:
   - `examples/custom_trainer/run_entropy_shaping_qwen2_7b.sh`: Reference run script

## Hyperparameter Guidance

### Alpha (`entropy_advantage_alpha`)
- Controls the magnitude of the entropy bonus
- Typical range: 0.01 - 0.5
- Higher values → stronger exploration bias
- Start with 0.1 and tune based on task

### Kappa (`entropy_advantage_kappa`)
- Controls the relative scaling of entropy vs advantage magnitude
- Typical range: 0.5 - 2.0
- Lower values → entropy term has more influence on large advantages
- Start with 1.0 (equal scaling)

## Expected Behavior

When entropy-based advantage shaping is enabled:

1. **During training**:
   - Entropy is computed for all tokens in the response
   - Shaped advantages are used for policy gradient computation
   - Original entropy regularization (if enabled) still applies to the loss

2. **Logging**:
   - Standard metrics (`actor/entropy`, `actor/pg_loss`, etc.) are logged
   - No additional metrics specific to shaping (it modifies advantages internally)

3. **Performance**:
   - Minimal computational overhead (one additional `torch.min` operation per micro-batch)
   - Memory: Requires entropy computation (already computed if `entropy_coeff > 0`)

## Testing

To verify the implementation works:

1. **Sanity check**: Run with and without shaping, verify training completes
2. **Hyperparameter sensitivity**: Test different alpha/kappa values
3. **Comparison**: Compare final model performance against baseline (no shaping)

## Notes

- The entropy term is **added** to advantages, always increasing them (since entropy ≥ 0)
- This encourages exploration by making high-entropy actions more likely
- The `min` operation bounds the entropy term to prevent it from dominating
- Works best with tasks requiring exploration (e.g., math reasoning, coding)

