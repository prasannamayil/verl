#!/bin/bash
# Wrapper script for mid-entropy sampling evaluation
# Usage: ./eval_checkpoint_mid_entropy.sh [checkpoint_path]

CHECKPOINT_PATH=${1}

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path required"
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 /fast/pmayilvahanan/verl_checkpoints/exploration/grpo_qwen25math_7b/global_step_160"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Call parametric script with mid-entropy settings
${SCRIPT_DIR}/eval_checkpoint_parametric.sh \
    "${CHECKPOINT_PATH}" \
    "mid_entropy" \
    1024 \
    0.6 \
    0.95 \
    -1 \
    "mid_entropy"

