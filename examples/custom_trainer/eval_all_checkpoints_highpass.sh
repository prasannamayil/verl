#!/bin/bash
set -x

# Script to evaluate all checkpoints with high pass@k
# Usage: ./eval_all_checkpoints_highpass.sh [checkpoint_dir] [high_pass_k]

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

module load cuda/12.9
source ~/.verl_102025/bin/activate

# Get checkpoint directory from command line or use default
CHECKPOINT_DIR=${1:-"/fast/pmayilvahanan/verl_checkpoints/exploration/gspo_qwen25math_1.5b_dapo_5k_epochs20_rollouts8_bsz1024_resp_len1024"}
HIGH_PASS_K=${2:-1024}

# Verify checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

echo "Evaluating all checkpoints in: $CHECKPOINT_DIR"
echo "High pass@k: $HIGH_PASS_K"

# Extract configuration from the checkpoint directory name or set defaults
# These should match the original training configuration
model_name=Qwen/Qwen2.5-Math-1.5B
project_name=verl_exploration

# Model configuration
response_length=1024
prompt_length=3072
total_ctx=$((prompt_length + response_length))

# High pass@k configuration
# For high pass@k evaluation, we need to:
# 1. Set rollout.n to the high pass@k value
# 2. Adjust validation batch size to avoid OOM
# With 175 test examples and pass@1024, we need smaller batches
# Assuming 8 GPUs with tensor_model_parallel_size=2, we have 4 effective workers
# We want batch_size such that 175 examples fit reasonably
# Let's use a smaller batch size for validation

# For 175 examples with pass@1024, we generate 175*1024 = 179,200 sequences total
# We need to process them in smaller batches to avoid OOM
# Let's use a validation batch size of 4 (so 4*1024=4096 sequences per batch)
# This will require ~44 batches to process all 175 examples
val_batch_size=4  # Number of prompts per validation batch

# For pass@1024, each prompt generates 1024 responses
# With val_batch_size=4, we process 4 prompts at a time = 4096 sequences
# Make sure log_prob_micro_batch_size_per_gpu can handle this
ppo_mini_batch_size=512
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=160

# GPU configuration
n_gpus=8
nnodes=1

# GSPO-specific configuration (must match training)
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
learning_rate=1e-6

# GSPO clipping parameters
clip_ratio_low=0.0003
clip_ratio_high=0.0004

# No KL for GSPO
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# Training dataset (needed even in val_only mode for dataloader initialization)
train_file=/fast/pmayilvahanan/datasets/dapo_math_17k/dapo_non_matching_math_b_5k.parquet

# Evaluation datasets (same as training)
val_files="[/fast/pmayilvahanan/datasets/aime_2024/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/aime_2025/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/math_b_exploration/qwen25math15_unsolved_no_suffix.parquet]"

# Extract experiment name from checkpoint directory
experiment_name=$(basename "$CHECKPOINT_DIR")

# Ensure results directory exists
results_dir=results
mkdir -p ${results_dir}

# Run the evaluation script
python3 -m verl.trainer.main_eval_all_checkpoints \
    +trainer.checkpoint_parent_dir=${CHECKPOINT_DIR} \
    +trainer.eval_filename=evals_high_pass.jsonl \
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
    actor_rollout_ref.rollout.n=${HIGH_PASS_K} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
    trainer.val_only=True \
    trainer.val_before_train=True \
    reward_model.reward_manager=dapo

echo ""
echo "================================================================================"
echo "Evaluation complete!"
echo "================================================================================"
echo "Results saved to: ${CHECKPOINT_DIR}/evals_high_pass.jsonl"
echo "Results copy saved to: ${results_dir}/${experiment_name}_evals_high_pass.jsonl"
echo "Validation traces saved to: ${CHECKPOINT_DIR}/validation_data_high_pass/"
echo "================================================================================"

