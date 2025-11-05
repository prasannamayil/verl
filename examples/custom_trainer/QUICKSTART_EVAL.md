# Quick Start: Evaluate Checkpoints with High Pass@k

## TL;DR

```bash
# Evaluate base model (before training) with pass@1024
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 1024

# Evaluate all checkpoints with pass@1024
./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir 1024

# Evaluate single checkpoint with pass@1024
./eval_single_checkpoint_highpass.sh /path/to/checkpoint_dir/global_step_80 1024

# View results
python view_eval_results.py /path/to/checkpoint_dir/evals_high_pass.jsonl --metrics pass@1024
```

## Examples

### Evaluate Base Model (Baseline)

```bash
cd /fast/pmayilvahanan/verl/examples/custom_trainer

# Evaluate base model before any training
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 1024

# Custom output directory
./eval_base_model_highpass.sh Qwen/Qwen2.5-Math-1.5B 1024 /path/to/output

# View base model results
python view_eval_results.py results/Qwen2_5-Math-1_5B_base_evals_base_pass1024.jsonl
```

### Evaluate Trained Checkpoints

```bash
cd /fast/pmayilvahanan/verl/examples/custom_trainer

./eval_all_checkpoints_highpass.sh \
    /fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024 \
    1024

# 2. View results (after evaluation completes)
python view_eval_results.py \
    /fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024/evals_high_pass.jsonl \
    --metrics pass@1024 pass@512
```

## What Gets Created

After running the evaluation, you'll have:

```
checkpoint_dir/
├── evals_high_pass.jsonl              # ✓ All metrics in one file
├── validation_data_high_pass/         # ✓ All traces saved
│   ├── 8.jsonl
│   ├── 16.jsonl
│   ├── 24.jsonl
│   └── ...
└── results/
    └── experiment_name_evals_high_pass.jsonl  # Copy for version control
```

## Key Features

✓ **Resume Support** - Automatically skips already-evaluated checkpoints  
✓ **High Pass@k** - Supports pass@1024, pass@512, etc.  
✓ **Saves Traces** - All validation generations are saved  
✓ **Batch Size Auto-adjust** - Prevents OOM with smaller batches  
✓ **Per-dataset Metrics** - Separate metrics for each evaluation dataset  

## Common Parameters to Adjust

### Change Pass@k Value
```bash
./eval_all_checkpoints_highpass.sh /path/to/checkpoint_dir 512  # pass@512 instead
```

### Adjust Batch Size (for OOM issues)
Edit `eval_all_checkpoints_highpass.sh`:
```bash
val_batch_size=2  # Reduce from default 4
```

### Force Re-evaluation
```bash
python3 -m verl.trainer.main_eval_all_checkpoints \
    +trainer.checkpoint_parent_dir=/path/to/checkpoint_dir \
    +trainer.eval_filename=evals_high_pass.jsonl \
    +trainer.force_reeval=True \
    [... other params ...]
```

**Note**: Use the `+` prefix for `checkpoint_parent_dir`, `eval_filename`, and `force_reeval` since these are new fields not in the base config.

## Expected Output

### During Evaluation
```
Checkpoint directory: /path/to/checkpoint_dir
Found 10 checkpoints
Already evaluated 5 checkpoints
Will evaluate 5 checkpoints

================================================================================
Evaluating checkpoint: /path/to/checkpoint_dir/global_step_48
Step: 48
================================================================================

[... evaluation logs ...]

✓ Successfully evaluated checkpoint at step 48

[... continues for remaining checkpoints ...]

================================================================================
Evaluation Summary
================================================================================
Total checkpoints: 5
Successful: 5
Failed: 0
================================================================================
```

### View Results
```
$ python view_eval_results.py checkpoint_dir/evals_high_pass.jsonl --metrics pass@1024

Reading evaluation results from: checkpoint_dir/evals_high_pass.jsonl
Found 10 evaluation entries
Found results for 10 checkpoints

Step       pass@1024                
--------------------------------------------------------------
8          0.342857                 
16         0.428571                 
24         0.485714                 
32         0.514286                 
40         0.542857                 
48         0.571429                 
56         0.600000                 
64         0.628571                 
72         0.657143                 
80         0.685714                 

=== Summary Statistics ===
Metric                                             Min          Max          Mean         Last        
----------------------------------------------------------------------------------------------------
pass@1024                                          0.342857     0.685714     0.539286     0.685714    
```

## Tips

- **Time Estimate**: ~30-60 minutes per checkpoint with pass@1024 (depends on dataset size)
- **Disk Space**: ~100-500 MB per checkpoint for traces
- **GPU Memory**: Reduce `val_batch_size` if OOM occurs
- **Interruption**: Safe to interrupt and restart - will resume from last completed checkpoint

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `val_batch_size` in script (e.g., to 2 or 1) |
| Slow Evaluation | Normal for high pass@k; consider evaluating fewer checkpoints |
| No Checkpoints Found | Verify checkpoint directory path and structure |
| Results Not Showing | Check that `evals_high_pass.jsonl` was created and has content |

## See Also

- `README_EVAL_CHECKPOINTS.md` - Comprehensive documentation
- `eval_single_checkpoint_highpass.sh` - Evaluate just one checkpoint
- `view_eval_results.py --help` - Full options for viewing results

