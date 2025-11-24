#!/bin/bash
# Wrapper script for high-entropy sampling evaluation of base model
# Usage: ./eval_base_model_high_entropy.sh [model_path] [run_id]

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

# Call parametric script with high-entropy settings
${SCRIPT_DIR}/eval_base_model_parametric.sh \
    "${MODEL_PATH}" \
    "high_entropy" \
    1024 \
    1.0 \
    1.0 \
    -1 \
    "high_entropy" \
    "${RUN_ID}"

