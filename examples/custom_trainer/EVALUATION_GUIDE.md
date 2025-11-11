# Multi-Strategy Evaluation Scripts

This directory contains scripts for evaluating checkpoints with different sampling strategies.

## Overview

Three sampling strategies are supported:

1. **Greedy Sampling** (n=1, temp=0.0)
   - Deterministic, picks most likely token
   - Single output per input

2. **Mid-Entropy Sampling** (n=1024, temp=0.6, top_p=0.95)
   - Moderate exploration
   - 1024 samples per input for pass@k metrics

3. **High-Entropy Sampling** (n=1024, temp=1.0, top_p=1.0)
   - Maximum diversity
   - 1024 samples per input for pass@k metrics

## Files

### Core Scripts

- **`eval_checkpoint_parametric.sh`**: Base script that accepts all sampling parameters
- **`eval_checkpoint_greedy.sh`**: Wrapper for greedy sampling
- **`eval_checkpoint_mid_entropy.sh`**: Wrapper for mid-entropy sampling  
- **`eval_checkpoint_high_entropy.sh`**: Wrapper for high-entropy sampling
- **`batch_eval_checkpoints.sh`**: Batch runner for multiple checkpoints

### Configuration Files

- **`checkpoint_list_example.txt`**: Template for checkpoint lists

## Usage

### Single Checkpoint Evaluation

#### Greedy Sampling
```bash
./eval_checkpoint_greedy.sh /path/to/checkpoint/global_step_160
```

#### Mid-Entropy Sampling
```bash
./eval_checkpoint_mid_entropy.sh /path/to/checkpoint/global_step_160
```

#### High-Entropy Sampling
```bash
./eval_checkpoint_high_entropy.sh /path/to/checkpoint/global_step_160
```

#### Custom Parameters
```bash
./eval_checkpoint_parametric.sh \
    /path/to/checkpoint/global_step_160 \
    custom_sampling \
    512 \
    0.7 \
    0.9 \
    -1 \
    custom_name
```

### Batch Evaluation

#### Create Checkpoint List

Create a text file with one checkpoint path per line:

```bash
cat > my_checkpoints.txt << EOF
/fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_160
/fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_170
/fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_180
EOF
```

#### Run All Sampling Strategies
```bash
./batch_eval_checkpoints.sh my_checkpoints.txt all
```

#### Run Specific Sampling Strategy
```bash
# Greedy only
./batch_eval_checkpoints.sh my_checkpoints.txt greedy

# Mid-entropy only
./batch_eval_checkpoints.sh my_checkpoints.txt mid_entropy

# High-entropy only
./batch_eval_checkpoints.sh my_checkpoints.txt high_entropy
```

## Output Files

For each checkpoint and sampling strategy, the following files are generated:

### In Checkpoint Directory

- `evals_{suffix}.jsonl`: Summary metrics (e.g., `evals_greedy.jsonl`, `evals_mid_entropy.jsonl`)
- `validation_data_{suffix}/`: Directory with detailed per-sample outputs

### In Results Directory

- `results/{experiment_name}_evals_{suffix}.jsonl`: Copy of evaluation metrics

### JSONL Format

Each validation output includes:

```json
{
  "input": "...",
  "output": "...", 
  "gts": "...",
  "score": 0.0,
  "step": 160,
  "uid": "uuid-string",
  "data_source": "aime_2024",
  "reward": 0.0,
  "acc": 0.0
}
```

**New in these scripts**: `data_source` field is now included in all validation outputs!

## Examples

### Evaluate Single Checkpoint with All Strategies

```bash
CHECKPOINT=/fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_160

./eval_checkpoint_greedy.sh $CHECKPOINT
./eval_checkpoint_mid_entropy.sh $CHECKPOINT
./eval_checkpoint_high_entropy.sh $CHECKPOINT
```

### Batch Evaluate Training Run

```bash
# List all checkpoints from a training run
ls -d /fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_* > checkpoints.txt

# Evaluate all with all strategies
./batch_eval_checkpoints.sh checkpoints.txt all
```

### Evaluate Multiple Training Runs

```bash
# Create list with checkpoints from different runs
cat > multi_run_checkpoints.txt << EOF
/fast/pmayilvahanan/verl_checkpoints/exploration/run1/global_step_160
/fast/pmayilvahanan/verl_checkpoints/exploration/run2/global_step_160
/fast/pmayilvahanan/verl_checkpoints/exploration/run3/global_step_160
EOF

./batch_eval_checkpoints.sh multi_run_checkpoints.txt all
```

## Configuration Notes

### Model and Dataset

Edit `eval_checkpoint_parametric.sh` to change:

- `model_name`: Base model path (default: `Qwen/Qwen2.5-Math-7B`)
- `train_file`: Training dataset (needed for dataloader init)
- `val_files`: Validation datasets (list of parquet files)

### Context Lengths

Current defaults:
- `prompt_length=1024`
- `response_length=3072`

### GPU Configuration

Current defaults:
- `n_gpus=8`
- `nnodes=1`

### Algorithm Parameters

GRPO-specific (modify in `eval_checkpoint_parametric.sh`):
- `adv_estimator=grpo`
- `loss_mode=grpo`
- `clip_ratio_low=0.0003`
- `clip_ratio_high=0.0004`

## Troubleshooting

### Checkpoint Not Found
```
Error: Checkpoint path does not exist: /path/to/checkpoint
```
**Solution**: Verify the checkpoint path exists and is accessible.

### CUDA Out of Memory

**Solution**: Reduce `val_batch_size` in `eval_checkpoint_parametric.sh`.

### Missing Dependencies

**Solution**: Ensure environment is activated:
```bash
source ~/.verl_102025/bin/activate
module load cuda/12.9
```

## Performance Notes

- **Greedy**: ~10-15 minutes per checkpoint (175 examples × 1 sample)
- **Mid/High-Entropy**: ~2-3 hours per checkpoint (175 examples × 1024 samples)
- **Batch "all" mode**: ~5-6 hours per checkpoint (all three strategies)

For large-scale evaluations, consider using job scheduling systems.

