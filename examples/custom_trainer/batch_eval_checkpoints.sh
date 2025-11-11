#!/bin/bash
set -e

# Batch evaluation script for multiple checkpoints
# Usage: ./batch_eval_checkpoints.sh [checkpoint_list_file] [sampling_type]
# 
# checkpoint_list_file: Text file with one checkpoint path per line
# sampling_type: greedy, mid_entropy, high_entropy, or all (default: all)
#
# Example checkpoint_list.txt:
# /fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_160
# /fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_170
# /fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_180

CHECKPOINT_LIST=${1}
SAMPLING_TYPE=${2:-"all"}

if [ -z "$CHECKPOINT_LIST" ]; then
    echo "Error: Checkpoint list file required"
    echo "Usage: $0 <checkpoint_list_file> [sampling_type]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_list_file: Text file with one checkpoint path per line"
    echo "  sampling_type: greedy, mid_entropy, high_entropy, or all (default: all)"
    echo ""
    echo "Example:"
    echo "  $0 checkpoint_list.txt all"
    echo ""
    echo "Example checkpoint_list.txt:"
    echo "  /path/to/checkpoint1/global_step_160"
    echo "  /path/to/checkpoint2/global_step_170"
    exit 1
fi

if [ ! -f "$CHECKPOINT_LIST" ]; then
    echo "Error: Checkpoint list file not found: $CHECKPOINT_LIST"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Count total checkpoints
total_checkpoints=$(grep -v '^\s*$' "$CHECKPOINT_LIST" | grep -v '^#' | wc -l)

echo "================================================================================"
echo "Batch Evaluation Starting"
echo "================================================================================"
echo "Checkpoint list: $CHECKPOINT_LIST"
echo "Total checkpoints: $total_checkpoints"
echo "Sampling type: $SAMPLING_TYPE"
echo "================================================================================"
echo ""

# Track overall progress
checkpoint_num=0
failed_checkpoints=()
batch_start_time=$(date +%s)

# Read checkpoint list and evaluate each
while IFS= read -r checkpoint_path; do
    # Skip empty lines and comments
    [[ -z "$checkpoint_path" ]] && continue
    [[ "$checkpoint_path" =~ ^[[:space:]]*# ]] && continue
    
    checkpoint_num=$((checkpoint_num + 1))
    
    echo ""
    echo "================================================================================"
    echo "Processing checkpoint $checkpoint_num/$total_checkpoints"
    echo "================================================================================"
    echo "Path: $checkpoint_path"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "WARNING: Checkpoint path does not exist: $checkpoint_path"
        echo "Skipping..."
        failed_checkpoints+=("$checkpoint_path (does not exist)")
        continue
    fi
    
    # Run evaluation(s) based on sampling type
    case "$SAMPLING_TYPE" in
        greedy)
            echo "Running greedy evaluation..."
            if ! ${SCRIPT_DIR}/eval_checkpoint_greedy.sh "$checkpoint_path"; then
                echo "ERROR: Greedy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (greedy failed)")
            fi
            ;;
        mid_entropy)
            echo "Running mid-entropy evaluation..."
            if ! ${SCRIPT_DIR}/eval_checkpoint_mid_entropy.sh "$checkpoint_path"; then
                echo "ERROR: Mid-entropy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (mid_entropy failed)")
            fi
            ;;
        high_entropy)
            echo "Running high-entropy evaluation..."
            if ! ${SCRIPT_DIR}/eval_checkpoint_high_entropy.sh "$checkpoint_path"; then
                echo "ERROR: High-entropy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (high_entropy failed)")
            fi
            ;;
        all)
            echo "Running all sampling strategies..."
            
            echo ""
            echo "--- Greedy Sampling ---"
            if ! ${SCRIPT_DIR}/eval_checkpoint_greedy.sh "$checkpoint_path"; then
                echo "ERROR: Greedy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (greedy failed)")
            fi
            
            echo ""
            echo "--- Mid-Entropy Sampling ---"
            if ! ${SCRIPT_DIR}/eval_checkpoint_mid_entropy.sh "$checkpoint_path"; then
                echo "ERROR: Mid-entropy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (mid_entropy failed)")
            fi
            
            echo ""
            echo "--- High-Entropy Sampling ---"
            if ! ${SCRIPT_DIR}/eval_checkpoint_high_entropy.sh "$checkpoint_path"; then
                echo "ERROR: High-entropy evaluation failed for $checkpoint_path"
                failed_checkpoints+=("$checkpoint_path (high_entropy failed)")
            fi
            ;;
        *)
            echo "ERROR: Unknown sampling type: $SAMPLING_TYPE"
            echo "Valid options: greedy, mid_entropy, high_entropy, all"
            exit 1
            ;;
    esac
    
    echo ""
    echo "Checkpoint $checkpoint_num/$total_checkpoints complete"
    
done < "$CHECKPOINT_LIST"

# Calculate total time
batch_end_time=$(date +%s)
batch_duration=$((batch_end_time - batch_start_time))
batch_hours=$((batch_duration / 3600))
batch_minutes=$(((batch_duration % 3600) / 60))
batch_seconds=$((batch_duration % 60))

# Print summary
echo ""
echo "================================================================================"
echo "Batch Evaluation Complete!"
echo "================================================================================"
echo "Total checkpoints processed: $checkpoint_num"
echo "Sampling type: $SAMPLING_TYPE"
echo "Total time: ${batch_hours}h ${batch_minutes}m ${batch_seconds}s"

if [ ${#failed_checkpoints[@]} -gt 0 ]; then
    echo ""
    echo "Failed checkpoints: ${#failed_checkpoints[@]}"
    for failed in "${failed_checkpoints[@]}"; do
        echo "  - $failed"
    done
    echo "================================================================================"
    exit 1
else
    echo "All checkpoints evaluated successfully!"
    echo "================================================================================"
fi

