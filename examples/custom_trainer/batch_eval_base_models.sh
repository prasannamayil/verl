#!/bin/bash
set -e

# Batch evaluation script for multiple base models
# Usage: ./batch_eval_base_models.sh [model_list_file] [sampling_type] [run_id]
# 
# model_list_file: Text file with one model path per line
# sampling_type: greedy, mid_entropy, high_entropy, or all (default: all)
# run_id: Optional run identifier (e.g., "run2", "seed42")
#
# Example model_list.txt:
# Qwen/Qwen2.5-Math-7B
# Qwen/Qwen2.5-Math-1.5B
# meta-llama/Llama-3.1-8B

MODEL_LIST=${1}
SAMPLING_TYPE=${2:-"all"}
RUN_ID=${3:-""}

if [ -z "$MODEL_LIST" ]; then
    echo "Error: Model list file required"
    echo "Usage: $0 <model_list_file> [sampling_type] [run_id]"
    echo ""
    echo "Arguments:"
    echo "  model_list_file: Text file with one model path per line"
    echo "  sampling_type: greedy, mid_entropy, high_entropy, or all (default: all)"
    echo "  run_id: Optional run identifier (e.g., 'run2', 'seed42')"
    echo ""
    echo "Example:"
    echo "  $0 model_list.txt all"
    echo "  $0 model_list.txt all run2"
    echo ""
    echo "Example model_list.txt:"
    echo "  Qwen/Qwen2.5-Math-7B"
    echo "  Qwen/Qwen2.5-Math-1.5B"
    exit 1
fi

if [ ! -f "$MODEL_LIST" ]; then
    echo "Error: Model list file not found: $MODEL_LIST"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Count total models
total_models=$(grep -v '^\s*$' "$MODEL_LIST" | grep -v '^#' | wc -l)

echo "================================================================================"
echo "Base Model Batch Evaluation Starting"
echo "================================================================================"
echo "Model list: $MODEL_LIST"
echo "Total models: $total_models"
echo "Sampling type: $SAMPLING_TYPE"
echo "================================================================================"
echo ""

if [ "$SAMPLING_TYPE" = "all" ]; then
    echo "Resume mode: enabled (skips sampling types already completed for a model)"
fi

# Track overall progress
model_num=0
failed_models=()
batch_start_time=$(date +%s)

# Helper to check if a base model already has results for a given suffix
is_completed_for_suffix() {
    local model_path="$1"
    local suffix="$2"
    local model_basename=$(basename "$model_path")
    local model_clean_name=$(echo "$model_basename" | tr '/' '_' | tr '.' '_')
    local parent_dir="/fast/pmayilvahanan/verl_checkpoints/base_models/${model_clean_name}_base"
    local experiment_name="${model_clean_name}_base"
    local primary_file="${parent_dir}/evals_${suffix}.jsonl"
    local repo_file="${SCRIPT_DIR}/../results/${experiment_name}_evals_${suffix}.jsonl"
    local trace_dir="${parent_dir}/validation_data_${suffix}"
    
    # Check if primary eval has step 0 entry and traces exist
    if [ -s "$primary_file" ] && grep -q "\"log_step\"[[:space:]]*:[[:space:]]*0\b" "$primary_file" && \
       [ -d "$trace_dir" ] && [ -n "$(ls -A "$trace_dir" 2>/dev/null)" ]; then
        return 0
    fi
    # Check if repo eval has step 0 entry and traces exist
    if [ -s "$repo_file" ] && grep -q "\"log_step\"[[:space:]]*:[[:space:]]*0\b" "$repo_file" && \
       [ -d "$trace_dir" ] && [ -n "$(ls -A "$trace_dir" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

# Read model list robustly (handles missing trailing newline)
while IFS= read -r model_path || [[ -n "$model_path" ]]; do
    # Skip empty lines and comments
    [[ -z "$model_path" ]] && continue
    [[ "$model_path" =~ ^[[:space:]]*# ]] && continue
    # Sanitize: strip CRs and trailing/leading whitespace
    model_path="${model_path//$'\r'/}"
    model_path="$(printf "%s" "$model_path" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    
    # Skip if empty after sanitization
    [[ -z "$model_path" ]] && continue
    
    model_num=$((model_num + 1))
    
    echo ""
    echo "================================================================================"
    echo "Processing model $model_num/$total_models"
    echo "================================================================================"
    echo "Model: $model_path"
    echo ""
    
    # Run evaluation(s) based on sampling type
    case "$SAMPLING_TYPE" in
        greedy)
            echo "Running greedy evaluation..."
            if ! ${SCRIPT_DIR}/eval_base_model_greedy.sh "$model_path" "$RUN_ID"; then
                echo "ERROR: Greedy evaluation failed for $model_path"
                failed_models+=("$model_path (greedy failed)")
            fi
            ;;
        mid_entropy)
            echo "Running mid-entropy evaluation..."
            if ! ${SCRIPT_DIR}/eval_base_model_mid_entropy.sh "$model_path" "$RUN_ID"; then
                echo "ERROR: Mid-entropy evaluation failed for $model_path"
                failed_models+=("$model_path (mid_entropy failed)")
            fi
            ;;
        high_entropy)
            echo "Running high-entropy evaluation..."
            if ! ${SCRIPT_DIR}/eval_base_model_high_entropy.sh "$model_path" "$RUN_ID"; then
                echo "ERROR: High-entropy evaluation failed for $model_path"
                failed_models+=("$model_path (high_entropy failed)")
            fi
            ;;
        all)
            echo "Running all sampling strategies (with resume)..."
            
            echo ""
            echo "--- Greedy Sampling ---"
            if is_completed_for_suffix "$model_path" "greedy"; then
                echo "SKIP: Greedy already completed for $model_path"
            else
                if ! ${SCRIPT_DIR}/eval_base_model_greedy.sh "$model_path" "$RUN_ID"; then
                    echo "ERROR: Greedy evaluation failed for $model_path"
                    failed_models+=("$model_path (greedy failed)")
                fi
            fi
            
            echo ""
            echo "--- Mid-Entropy Sampling ---"
            if is_completed_for_suffix "$model_path" "mid_entropy"; then
                echo "SKIP: Mid-entropy already completed for $model_path"
            else
                if ! ${SCRIPT_DIR}/eval_base_model_mid_entropy.sh "$model_path" "$RUN_ID"; then
                    echo "ERROR: Mid-entropy evaluation failed for $model_path"
                    failed_models+=("$model_path (mid_entropy failed)")
                fi
            fi
            
            echo ""
            echo "--- High-Entropy Sampling ---"
            if is_completed_for_suffix "$model_path" "high_entropy"; then
                echo "SKIP: High-entropy already completed for $model_path"
            else
                if ! ${SCRIPT_DIR}/eval_base_model_high_entropy.sh "$model_path" "$RUN_ID"; then
                    echo "ERROR: High-entropy evaluation failed for $model_path"
                    failed_models+=("$model_path (high_entropy failed)")
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
    echo "Model $model_num/$total_models complete"
    
done < "$MODEL_LIST"

# Calculate total time
batch_end_time=$(date +%s)
batch_duration=$((batch_end_time - batch_start_time))
batch_hours=$((batch_duration / 3600))
batch_minutes=$(((batch_duration % 3600) / 60))
batch_seconds=$((batch_duration % 60))

# Print summary
echo ""
echo "================================================================================"
echo "Base Model Batch Evaluation Complete!"
echo "================================================================================"
echo "Total models processed: $model_num"
echo "Sampling type: $SAMPLING_TYPE"
echo "Total time: ${batch_hours}h ${batch_minutes}m ${batch_seconds}s"

if [ ${#failed_models[@]} -gt 0 ]; then
    echo ""
    echo "Failed models: ${#failed_models[@]}"
    for failed in "${failed_models[@]}"; do
        echo "  - $failed"
    done
    echo "================================================================================"
    exit 1
else
    echo "All models evaluated successfully!"
    echo "================================================================================"
fi

