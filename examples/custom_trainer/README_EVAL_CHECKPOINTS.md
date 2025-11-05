# High Pass@k Checkpoint Evaluation

This directory contains scripts for evaluating all checkpoints in a training run with high pass@k (e.g., pass@1024).

## Overview

The evaluation system supports:
- **Base model evaluation** - evaluate untrained model for baseline metrics
- **Automatic evaluation of all checkpoints** in a directory
- **Resumable evaluation** - skips already-evaluated checkpoints
- **High pass@k metrics** (pass@1024, pass@512, etc.)
- **Automatic batch size adjustment** to avoid OOM
- **Saves evaluation traces** for each checkpoint
- **Separate result files** (`evals_high_pass.jsonl`) to not interfere with training logs
- **Timing tracking** - records evaluation duration for each checkpoint

## Files

- `eval_base_model_highpass.sh` - **NEW**: Evaluate base model (before training) for baseline
- `eval_all_checkpoints_highpass.sh` - Evaluate all checkpoints with high pass@k
- `eval_single_checkpoint_highpass.sh` - Evaluate a single checkpoint
- `view_eval_results.py` - View and analyze evaluation results
- `README_EVAL_CHECKPOINTS.md` - This file
- `QUICKSTART_EVAL.md` - Quick start guide

## Quick Start

### Evaluate Base Model (Baseline)

```bash
# Evaluate base model before training (pass@1024)
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 1024

# Custom pass@k and output directory
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 512 /path/to/output
```

**Creates**:
- `/path/to/output/Qwen2_5-Math-1_5B_pass1024/evals_base_pass1024.jsonl`
- `/path/to/output/Qwen2_5-Math-1_5B_pass1024/validation_data_base_pass1024/`
- `results/Qwen2_5-Math-1_5B_base_evals_base_pass1024.jsonl`

### Evaluate All Checkpoints

```bash
# Basic usage with defaults (pass@1024)
./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir

# Custom pass@k value
./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir 512
```

### Evaluate a Single Checkpoint

```bash
# Evaluate specific checkpoint
./eval_single_checkpoint_highpass.sh /path/to/checkpoint_dir/global_step_48 1024
```

### View Results

```bash
# View all metrics
python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl

# View specific metrics
python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl --metrics pass@1024

# Summary only
python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl --summary-only
```

## How It Works

### 1. Checkpoint Discovery
The script automatically finds all `global_step_*` directories in the checkpoint folder.

### 2. Resume Support
Before evaluating, it checks `evals_high_pass.jsonl` to see which checkpoints have already been evaluated. Only unevaluated checkpoints are processed.

### 3. Batch Size Adjustment
For high pass@k (e.g., 1024 samples per prompt), the validation batch size is automatically reduced to avoid OOM:
- **Default validation batch size**: 4 prompts per batch
- **With pass@1024**: 4 * 1024 = 4,096 sequences per batch
- **With 175 test examples**: ~44 batches total

### 4. Output Files

After evaluation, you'll find:

```
checkpoint_dir/
├── evals_high_pass.jsonl          # Metrics for all evaluated checkpoints
├── validation_data_high_pass/     # Validation traces
│   ├── 8.jsonl                    # Traces for step 8
│   ├── 16.jsonl                   # Traces for step 16
│   └── ...
└── results/                       # Repo-level copies
    └── experiment_name_evals_high_pass.jsonl
```

### 5. Evaluation Metrics

The system computes various pass@k metrics:
- `pass@1`, `pass@2`, `pass@4`, `pass@8`, ..., `pass@1024`
- Mean/best/majority vote metrics for each dataset
- Per-dataset breakdowns (AIME 2024, AIME 2025, Math B, etc.)

## Configuration

### Adjust Validation Batch Size

Edit the script to change `val_batch_size`:

```bash
# In eval_all_checkpoints_highpass.sh
val_batch_size=4  # Increase if you have more GPU memory
```

**Guidelines**:
- **More GPU memory**: Increase `val_batch_size` (e.g., 8, 16)
- **Less GPU memory**: Decrease `val_batch_size` (e.g., 2)
- **Formula**: Total sequences per batch = `val_batch_size * rollout.n`

### Adjust Pass@k Value

```bash
# Pass@512 instead of pass@1024
./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir 512
```

### Change Evaluation Datasets

Edit the `val_files` parameter in the script:

```bash
val_files="[/path/to/dataset1.parquet,/path/to/dataset2.parquet]"
```

## Advanced Usage

### Force Re-evaluation

The Python script supports force re-evaluation:

```bash
python -m verl.trainer.main_eval_all_checkpoints \
    +trainer.checkpoint_parent_dir=/path/to/checkpoint_dir \
    +trainer.eval_filename=evals_high_pass.jsonl \
    +trainer.force_reeval=True \
    [... other params ...]
```

**Note**: The `+` prefix is required for `checkpoint_parent_dir`, `eval_filename`, and `force_reeval` since these are new fields not defined in the base trainer config.

### Custom Eval Filename

Use a different filename for results:

```bash
python -m verl.trainer.main_eval_all_checkpoints \
    +trainer.checkpoint_parent_dir=/path/to/checkpoint_dir \
    +trainer.eval_filename=evals_pass512.jsonl \
    [... other params ...]
```

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter OOM errors:

1. **Reduce validation batch size** in the script:
   ```bash
   val_batch_size=2  # or even 1
   ```

2. **Reduce pass@k value**:
   ```bash
   ./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir 512
   ```

3. **Adjust GPU memory utilization**:
   ```bash
   actor_rollout_ref.rollout.gpu_memory_utilization=0.7  # default is 0.8
   ```

### Evaluation Hangs

If evaluation appears to hang:
- Check GPU utilization with `nvidia-smi`
- Check Ray dashboard for worker status
- Look for error messages in the console output

### Missing Checkpoints

If some checkpoints are not found:
- Verify the checkpoint directory structure
- Check that checkpoints are named `global_step_N`
- Ensure checkpoints contain both `actor/` subdirectories

## Performance Tips

### Parallel Evaluation

For faster evaluation, you can run multiple evaluation jobs in parallel on different checkpoint directories or with different configurations.

### Disk Space

High pass@k evaluation generates large trace files:
- Each trace file: ~10-50 MB per checkpoint (depends on response length)
- With 10 checkpoints and pass@1024: ~500 MB - 5 GB total

Monitor disk space and clean up old traces if needed.

## Example Workflow

```bash
# 1. Evaluate base model (baseline)
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 1024

# 2. Train your model (generates checkpoints)
./run_gspo_qwen25math_1.5b.sh

# 3. Evaluate all checkpoints with pass@1024
./eval_all_checkpoints_highpass.sh \
    /fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024 \
    1024

# 4. Compare base model vs trained checkpoints
echo "=== Base Model ==="
python view_eval_results.py \
    results/Qwen2_5-Math-1_5B_base_evals_base_pass1024.jsonl \
    --metrics pass@1024 --summary-only

echo "=== Trained Checkpoints ==="
python view_eval_results.py \
    /fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024/evals_high_pass.jsonl \
    --metrics pass@1024 pass@512 pass@256

# 5. (Optional) Evaluate specific checkpoints with different pass@k
./eval_single_checkpoint_highpass.sh \
    /fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024/global_step_48 \
    512
```

## Implementation Details

### File Structure

The implementation consists of:
- `main_eval_all_checkpoints.py` - Python script to orchestrate evaluation
- Modified `ray_trainer.py` - Supports custom eval filenames and append mode
- Modified `json_logger.py` - Supports append mode for continuous evaluation

### Key Features

1. **Checkpoint Iteration**: Automatically discovers and sorts checkpoints by step number
2. **Resume Detection**: Reads existing `evals_high_pass.jsonl` to skip evaluated checkpoints
3. **Append Mode**: New evaluations are appended to existing files without overwriting
4. **Trace Naming**: Validation traces are saved with checkpoint-specific names
5. **Separate Directories**: High pass@k traces go to `validation_data_high_pass/` to avoid conflicts

## Notes

- **Training Configuration**: The evaluation scripts use the same model and dataset configurations as training
- **Sampling Parameters**: Temperature, top_p, and top_k match training rollout settings
- **Determinism**: Results may vary slightly between runs due to stochastic sampling
- **Resource Usage**: High pass@k evaluation is computationally expensive and may take several hours

## Support

For issues or questions:
1. Check this README for common solutions
2. Review the script comments for parameter descriptions
3. Check console output for error messages
4. Verify GPU memory and disk space availability

