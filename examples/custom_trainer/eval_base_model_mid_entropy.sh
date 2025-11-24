#!/bin/bash
# Wrapper script for mid-entropy sampling evaluation of base model
# Usage: ./eval_base_model_mid_entropy.sh [model_path] [run_id]

MODEL_PATH=${1}
RUN_ID=${2:-""}

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path required"
    echo "Usage: $0 <model_path> [run_id]"
    echo "Example: $0 Qwen/Qwen2.5-Math-7B"
    echo "Example: $0 Qwen/Qwen2.5-Math-7B run2"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Call parametric script with mid-entropy settings
${SCRIPT_DIR}/eval_base_model_parametric.sh \
    "${MODEL_PATH}" \
    "mid_entropy" \
    1024 \
    0.6 \
    0.95 \
    -1 \
    "mid_entropy" \
    "${RUN_ID}"

