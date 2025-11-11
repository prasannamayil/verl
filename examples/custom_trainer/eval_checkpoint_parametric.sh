#!/bin/bash
set -x

# Parameterized evaluation script for different sampling strategies
# Usage: ./eval_checkpoint_parametric.sh [checkpoint_path] [sampling_type] [n_samples] [temperature] [top_p] [top_k] [suffix]

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

module load cuda/12.9
source ~/.verl_102025/bin/activate

# Get parameters from command line
CHECKPOINT_PATH=${1}
SAMPLING_TYPE=${2:-"high_entropy"}  # greedy, mid_entropy, high_entropy
N_SAMPLES=${3:-1024}
TEMPERATURE=${4:-1.0}
TOP_P=${5:-1.0}
TOP_K=${6:--1}
SUFFIX=${7:-"${SAMPLING_TYPE}"}

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path required"
    echo "Usage: $0 <checkpoint_path> [sampling_type] [n_samples] [temperature] [top_p] [top_k] [suffix]"
    echo "Example: $0 /path/to/checkpoint greedy 1 0.0 1.0 -1 greedy"
    echo "Example: $0 /path/to/checkpoint mid_entropy 1024 0.6 0.95 -1 mid_entropy"
    echo "Example: $0 /path/to/checkpoint high_entropy 1024 1.0 1.0 -1 high_entropy"
    exit 1
fi

# Verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Extract parent directory for saving results
PARENT_DIR=$(dirname "$CHECKPOINT_PATH")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_PATH")

echo "================================================================================"
echo "Evaluation Configuration"
echo "================================================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Sampling Type: $SAMPLING_TYPE"
echo "N Samples: $N_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Suffix: $SUFFIX"
echo "Results will be saved to: $PARENT_DIR"
echo "================================================================================"

# Model configuration (should match training)
model_name=Qwen/Qwen2.5-Math-7B
project_name=verl_exploration

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

# GRPO-specific configuration (must match training)
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

# Extract experiment name from parent directory
experiment_name=$(basename "$PARENT_DIR")

# Ensure results directory exists
results_dir=results
mkdir -p ${results_dir}

# Set do_sample based on sampling type
if [ "$SAMPLING_TYPE" = "greedy" ]; then
    do_sample=False
else
    do_sample=True
fi

# Track evaluation time
eval_start_time=$(date +%s)

# Run evaluation
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
    actor_rollout_ref.model.path=${model_name} \
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
    trainer.resume_mode=resume_path \
    trainer.resume_from_path=${CHECKPOINT_PATH} \
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

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "Sampling: ${SAMPLING_TYPE} (n=${N_SAMPLES}, temp=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K})"
echo "Evaluation time: ${eval_minutes}m ${eval_seconds}s"
echo "-------------------------------------------------------------------------------"
echo "Results saved to: $PARENT_DIR/evals_${SUFFIX}.jsonl"
echo "Results copy saved to: ${results_dir}/${experiment_name}_evals_${SUFFIX}.jsonl"
echo "Traces saved to: $PARENT_DIR/validation_data_${SUFFIX}/${CHECKPOINT_NAME}.jsonl"
echo "================================================================================"

