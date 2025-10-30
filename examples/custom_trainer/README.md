# Custom Trainer Scripts for GRPO and DAPO

This directory contains runnable shell scripts and Condor submission files for running GRPO and DAPO training on your cluster.

## Directory Structure

```
custom_trainer/
├── README.md                       # This file
├── run_grpo_qwen25_7b.sh         # GRPO training for Qwen2.5-7B
├── run_grpo_qwen2_7b.sh          # GRPO training for Qwen2-7B
├── run_dapo_qwen25_7b.sh         # DAPO training for Qwen2.5-7B
├── run_dapo_qwen2_7b.sh          # DAPO training for Qwen2-7B
├── run_gspo_qwen25_7b.sh         # GSPO training for Qwen2.5-7B
├── run_gspo_qwen2_7b.sh          # GSPO training for Qwen2-7B
├── grpo_training.sub              # Condor submission for GRPO jobs
├── dapo_training.sub              # Condor submission for DAPO jobs
├── multi_job_training.sub         # Multi-job parallel execution
└── run_experiment.sh              # Wrapper for multi-job execution
```

## Shell Scripts

### GRPO Scripts
- **run_grpo_qwen25_7b.sh**: Runs GRPO training with Qwen2.5-7B-Instruct
  - Uses KL penalty (kl_coef=0.001)
  - 5 rollouts, 5 epochs by default
  - Batch size: 1024

- **run_grpo_qwen2_7b.sh**: Runs GRPO training with Qwen2-7B-Instruct
  - Similar configuration with adjusted batch sizes for Qwen2

### DAPO Scripts
- **run_dapo_qwen25_7b.sh**: Runs DAPO training with Qwen2.5-7B-Instruct
  - Uses reward_manager=dapo for Dense-and-Partial Off-policy alignment
  - No KL penalty (kl_coef=0.0)
  - 5 rollouts, 3 epochs by default
  - Smaller batch size: 512

- **run_dapo_qwen2_7b.sh**: Runs DAPO training with Qwen2-7B-Instruct
  - Similar DAPO configuration for Qwen2

### GSPO Scripts
- **run_gspo_qwen25_7b.sh**: Runs GSPO training with Qwen2.5-7B-Instruct
  - Uses gspo loss mode with sequence-level importance sampling
  - Special clipping ratios (clip_ratio_low=0.0003, clip_ratio_high=0.0004)
  - 16 rollouts, longer response length (8k tokens)
  - No KL penalty

- **run_gspo_qwen2_7b.sh**: Runs GSPO training with Qwen2-7B-Instruct
  - Similar GSPO configuration for Qwen2

## Configuration Parameters

All scripts support the following key parameters (can be modified in the scripts):

- `dataset`: Dataset to use (default: gsm8k)
- `rollouts`: Number of rollouts for sampling (default: 5)
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `response_length`: Maximum response length (default: 1024)
- `prompt_length`: Maximum prompt length (default: 512)
- `n_gpus`: Number of GPUs per node (default: 8)

## Running the Scripts

### Direct Execution
```bash
# Make scripts executable (already done)
chmod +x *.sh

# Run GRPO training
./run_grpo_qwen25_7b.sh

# Run DAPO training
./run_dapo_qwen25_7b.sh
```

### Using Condor Submission

#### Single Job Submission
```bash
# Submit GRPO job
condor_submit grpo_training.sub

# Submit DAPO job
condor_submit dapo_training.sub
```

#### Multi-Job Parallel Execution
```bash
# Submit 16 parallel jobs (4 each for GRPO/DAPO with both models)
condor_submit multi_job_training.sub
```

## Customizing the Scripts

### Changing Models
Edit the `model_name` variable in the shell scripts:
```bash
model_name=Qwen/Qwen2.5-7B-Instruct  # or any other model
```

### Adjusting Hyperparameters
Modify the configuration variables at the top of each script:
```bash
learning_rate=1e-6
kl_coef=0.001  # Set to 0.0 for DAPO
batch_size=1024
```

### Changing Dataset
Modify the `dataset` variable and ensure the parquet files exist:
```bash
dataset=gsm8k  # or your custom dataset
```

### Modifying Condor Requirements
Edit the `.sub` files to change:
- GPU requirements (e.g., A100-80GB, H100)
- Memory allocation
- CPU count
- Node exclusions

## Output Locations

- **Checkpoints**: `/fast/pmayilvahanan/verl_checkpoints/custom_trainer/`
- **Logs**: `/fast/pmayilvahanan/verl_jobs/{grpo,dapo,multi}_custom/`
- **Wandb**: Projects named `verl_grpo_custom` or `verl_dapo_custom`

## Key Differences: GRPO vs DAPO vs GSPO

### GRPO (Group Relative Policy Optimization)
- Uses KL penalty (kl_coef=0.001) to prevent policy drift
- Standard batch size (1024)
- 5 epochs, 5 rollouts
- Standard PPO-style optimization

### DAPO (Dense-and-Partial Off-policy)
- No KL penalty (kl_coef=0.0)
- Uses reward_manager=dapo for specialized reward computation
- Smaller batch size (512)
- 3 epochs, 5 rollouts
- Better for off-policy learning scenarios

### GSPO (Generalized Sequence Policy Optimization)
- No KL penalty (kl_coef=0.0)
- Uses special loss_mode=gspo with sequence-level importance sampling
- Asymmetric clipping (clip_ratio_low=0.0003, clip_ratio_high=0.0004)
- Much longer response length (8k tokens)
- 16 rollouts for better diversity
- 10 epochs for convergence

## Notes

- All scripts use VLLM for generation with tensor parallelism (TP=2)
- Gradient checkpointing is enabled to save memory
- Scripts save checkpoints based on configuration:
  - GRPO/DAPO: Every epoch (`save_freq=1`)
  - GSPO: Every 10 steps (`save_freq=10`)
- Multiple loggers are supported:
  - `console`: Console output
  - `wandb`: Weights & Biases tracking
  - `local_json`: Local JSON file logging (metrics saved to checkpoint directory)

## Advanced Features

### Local JSON Logging
The `local_json` logger has been added to save metrics locally:
- Metrics are saved to `{checkpoint_dir}/metrics.jsonl`
- Each line is a JSON object with timestamp, step, and metrics
- Useful for offline analysis and when wandb is unavailable

### Reward Managers
- **dapo**: Dense-and-Partial Off-policy reward manager
- Can be extended with overlong buffer configurations

### Loss Modes
- **ppo**: Standard PPO loss
- **gspo**: Generalized Sequence Policy Optimization with importance sampling

## Future Additions

Features that can be added later:
- Advanced advantage tracking and saving
- Prioritization strategies for sample selection
- Optimal trace selection for improved learning
- Custom advantage thresholds and filtering
