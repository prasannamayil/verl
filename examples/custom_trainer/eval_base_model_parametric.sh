#!/bin/bash
set -Eeuo pipefail
set -x

# Parameterized base model evaluation script for different sampling strategies
# Usage: ./eval_base_model_parametric.sh [model_path] [sampling_type] [n_samples] [temperature] [top_p] [top_k] [suffix]

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

module load cuda/12.9
source ~/.verl_102025/bin/activate

# Get the directory where this script is located (used to anchor paths)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get parameters from command line
MODEL_PATH=${1}
SAMPLING_TYPE=${2:-"high_entropy"}  # greedy, mid_entropy, high_entropy
N_SAMPLES=${3:-1024}
TEMPERATURE=${4:-1.0}
TOP_P=${5:-1.0}
TOP_K=${6:--1}
SUFFIX=${7:-"${SAMPLING_TYPE}"}
RUN_ID=${8:-""}  # Optional run identifier (e.g., "run1", "seed42")

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path required"
    echo "Usage: $0 <model_path> [sampling_type] [n_samples] [temperature] [top_p] [top_k] [suffix]"
    echo "Example: $0 Qwen/Qwen2.5-Math-7B greedy 1 0.0 1.0 -1 greedy"
    echo "Example: $0 Qwen/Qwen2.5-Math-7B mid_entropy 1024 0.6 0.95 -1 mid_entropy"
    echo "Example: $0 Qwen/Qwen2.5-Math-7B high_entropy 1024 1.0 1.0 -1 high_entropy"
    exit 1
fi

# Extract a clean name for the model (e.g., "Qwen2.5-Math-7B" from "Qwen/Qwen2.5-Math-7B")
model_basename=$(basename "$MODEL_PATH")
model_clean_name=$(echo "$model_basename" | tr '/' '_' | tr '.' '_')

# Create output directory structure similar to checkpoints
# Format: base_models/<model_name>_base[_runid]/
BASE_MODELS_DIR="/scratch/pmayilvahanan/verl_checkpoints/base_models"
if [ -n "$RUN_ID" ]; then
    PARENT_DIR="${BASE_MODELS_DIR}/${model_clean_name}_base_${RUN_ID}"
    experiment_name_suffix="_${RUN_ID}"
else
    PARENT_DIR="${BASE_MODELS_DIR}/${model_clean_name}_base"
    experiment_name_suffix=""
fi
mkdir -p "${PARENT_DIR}"

echo "================================================================================"
echo "Base Model Evaluation Configuration"
echo "================================================================================"
echo "Model: $MODEL_PATH"
echo "Sampling Type: $SAMPLING_TYPE"
echo "N Samples: $N_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Suffix: $SUFFIX"
echo "Results will be saved to: $PARENT_DIR"
echo "================================================================================"

# Project configuration
project_name=verl_exploration
experiment_name="${model_clean_name}_base${experiment_name_suffix}"

# Context configuration
response_length=3072
prompt_length=1024
total_ctx=$((prompt_length + response_length))

# Validation batch size - adjust based on n_samples
if [ "$N_SAMPLES" -eq 1 ]; then
    val_batch_size=16  # Greedy sampling can handle more at once
else
    val_batch_size=8   # High pass@k needs smaller batches
fi

# GPU and batch configuration
ppo_mini_batch_size=512
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=160
n_gpus=8
nnodes=1

# GRPO-specific configuration (for config compatibility)
adv_estimator=grpo
loss_mode=grpo
loss_agg_mode="seq-mean-token-mean"
learning_rate=1e-6

# GRPO clipping parameters
clip_ratio_low=0.0003
clip_ratio_high=0.0004

# No KL for GRPO
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# Training dataset (needed even in val_only mode for dataloader initialization)
train_file=/fast/pmayilvahanan/datasets/dapo_math_17k/dapo_non_matching_math_b_5k.parquet

# Evaluation datasets
val_files="[/fast/pmayilvahanan/datasets/aime_2024/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/aime_2025/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/math_b_exploration/qwen25math15_unsolved_no_suffix.parquet]"

# Ensure results directory exists
# Anchor repo-level results to repo root (examples/custom_trainer/..)
results_dir="${SCRIPT_DIR}/../results"
mkdir -p "${results_dir}"

# Set do_sample based on sampling type
if [ "$SAMPLING_TYPE" = "greedy" ]; then
    do_sample=False
else
    do_sample=True
fi

# Track evaluation time
eval_start_time=$(date +%s)

# Run evaluation (NO checkpoint loading - just base model)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=${total_ctx} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${total_ctx} \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${do_sample} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    data.train_files=${train_file} \
    data.val_files=${val_files} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','local_json'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    +trainer.repo_logs_dir=${results_dir} \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=${nnodes} \
    trainer.default_local_dir=${PARENT_DIR} \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.val_before_train=True \
    +trainer.eval_filename=evals_${SUFFIX}.jsonl \
    +trainer.validation_data_dirname=validation_data_${SUFFIX} \
    reward_model.reward_manager=dapo

# Calculate evaluation time
eval_end_time=$(date +%s)
eval_duration=$((eval_end_time - eval_start_time))
eval_minutes=$((eval_duration / 60))
eval_seconds=$((eval_duration % 60))

# Verify outputs exist and contain results
# For base model, we use step 0 as the identifier
primary_eval_file="$PARENT_DIR/evals_${SUFFIX}.jsonl"
repo_eval_file="${results_dir}/${experiment_name}_evals_${SUFFIX}.jsonl"
trace_dir="$PARENT_DIR/validation_data_${SUFFIX}"

# Helper: check if a jsonl contains step 0 entry
has_step_entry() {
    local f="$1"
    [ -s "$f" ] && grep -q "\"log_step\"[[:space:]]*:[[:space:]]*0\b" "$f"
}

primary_ok=false
repo_ok=false
trace_ok=false

if has_step_entry "$primary_eval_file"; then
    primary_ok=true
fi
if has_step_entry "$repo_eval_file"; then
    repo_ok=true
fi
if [ -d "$trace_dir" ] && [ -n "$(ls -A "$trace_dir" 2>/dev/null)" ]; then
    trace_ok=true
fi

if { [ "$primary_ok" = true ] || [ "$repo_ok" = true ]; } && [ "$trace_ok" = true ]; then
    echo ""
    echo "================================================================================"
    echo "Base model evaluation complete!"
    echo "================================================================================"
    echo "Model: ${MODEL_PATH}"
    echo "Sampling: ${SAMPLING_TYPE} (n=${N_SAMPLES}, temp=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K})"
    echo "Evaluation time: ${eval_minutes}m ${eval_seconds}s"
    echo "-------------------------------------------------------------------------------"
    echo "Results saved to: $primary_eval_file"
    echo "Results copy saved to: $repo_eval_file"
    echo "Traces saved to: $trace_dir"
    echo "================================================================================"
else
    echo ""
    echo "ERROR: Evaluation did not produce valid outputs."
    echo "       Checked:"
    echo "         - primary eval: $primary_eval_file (found_step=$primary_ok)"
    echo "         - repo eval:    $repo_eval_file (found_step=$repo_ok)"
    echo "         - trace dir:    $trace_dir (exists_nonempty=$trace_ok)"
    exit 1
fi

