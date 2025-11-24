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
if [ "$SAMPLING_TYPE" = "all" ]; then
    echo "Resume mode: enabled (skips sampling types already completed for a checkpoint)"
fi

# Helper to check if a checkpoint already has results for a given suffix
is_completed_for_suffix() {
    local checkpoint_path="$1"
    local suffix="$2"
    local checkpoint_name="$(basename "$checkpoint_path")"
    local parent_dir="$(dirname "$checkpoint_path")"
    local step="${checkpoint_name#global_step_}"
    local experiment_name="$(basename "$parent_dir")"
    local primary_file="${parent_dir}/evals_${suffix}.jsonl"
    local safe_hash
    safe_hash=$(echo -n "$experiment_name" | sha1sum | awk '{print substr($1,1,12)}')
    local repo_file="${SCRIPT_DIR}/../results/exp_${safe_hash}/evals_${suffix}.jsonl"
    local trace_file="${parent_dir}/validation_data_${suffix}/${checkpoint_name}.jsonl"
    # Require a non-empty trace file to consider completion
    if [ ! -s "$trace_file" ]; then
        return 1
    fi
    if [ -s "$primary_file" ] && grep -q "\"log_step\"[[:space:]]*:[[:space:]]*${step}\\b" "$primary_file"; then
        return 0
    fi
    if [ -s "$repo_file" ] && grep -q "\"log_step\"[[:space:]]*:[[:space:]]*${step}\\b" "$repo_file"; then
        return 0
    fi
    return 1
}

# Track overall progress
checkpoint_num=0
failed_checkpoints=()
batch_start_time=$(date +%s)

# Read checkpoint list robustly (handles missing trailing newline)
while IFS= read -r checkpoint_path || [[ -n "$checkpoint_path" ]]; do
    # Skip empty lines and comments
    [[ -z "$checkpoint_path" ]] && continue
    [[ "$checkpoint_path" =~ ^[[:space:]]*# ]] && continue
    # Sanitize: strip CRs and trailing slashes/spaces
    checkpoint_path="${checkpoint_path//$'\r'/}"
    checkpoint_path="${checkpoint_path%/}"
    # Trim trailing spaces
    checkpoint_path="$(printf "%s" "$checkpoint_path" | sed 's/[[:space:]]*$//')"
    
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
            echo "Running all sampling strategies (with resume)..."
            
            echo ""
            echo "--- Greedy Sampling ---"
            if is_completed_for_suffix "$checkpoint_path" "greedy"; then
                echo "SKIP: Greedy already completed for $checkpoint_path"
            else
                if ! ${SCRIPT_DIR}/eval_checkpoint_greedy.sh "$checkpoint_path"; then
                    echo "ERROR: Greedy evaluation failed for $checkpoint_path"
                    failed_checkpoints+=("$checkpoint_path (greedy failed)")
                fi
            fi
            
            echo ""
            echo "--- Mid-Entropy Sampling ---"
            if is_completed_for_suffix "$checkpoint_path" "mid_entropy"; then
                echo "SKIP: Mid-entropy already completed for $checkpoint_path"
            else
                if ! ${SCRIPT_DIR}/eval_checkpoint_mid_entropy.sh "$checkpoint_path"; then
                    echo "ERROR: Mid-entropy evaluation failed for $checkpoint_path"
                    failed_checkpoints+=("$checkpoint_path (mid_entropy failed)")
                fi
            fi
            
            echo ""
            echo "--- High-Entropy Sampling ---"
            if is_completed_for_suffix "$checkpoint_path" "high_entropy"; then
                echo "SKIP: High-entropy already completed for $checkpoint_path"
            else
                if ! ${SCRIPT_DIR}/eval_checkpoint_high_entropy.sh "$checkpoint_path"; then
                    echo "ERROR: High-entropy evaluation failed for $checkpoint_path"
                    failed_checkpoints+=("$checkpoint_path (high_entropy failed)")
                fi
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

