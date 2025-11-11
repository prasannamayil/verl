#!/bin/bash
set -x

# Script to evaluate base model (no training) with high pass@k
# Usage: ./eval_base_model_highpass.sh [model_name] [high_pass_k] [output_dir]

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

module load cuda/12.9
source ~/.verl_102025/bin/activate

# Get model name from command line or use default
MODEL_NAME=${1:-"Qwen/Qwen2.5-Math-1.5B"}
HIGH_PASS_K=${2:-1024}
OUTPUT_DIR=${3:-"/fast/pmayilvahanan/verl_checkpoints/base_models"}

echo "Evaluating base model: $MODEL_NAME"
echo "High pass@k: $HIGH_PASS_K"
echo "Output directory: $OUTPUT_DIR"

# Extract a clean name for the model (e.g., "Qwen2.5-Math-1.5B" from "Qwen/Qwen2.5-Math-1.5B")
model_basename=$(basename "$MODEL_NAME")
model_clean_name=$(echo "$model_basename" | tr '/' '_' | tr '.' '_')

# Create output directory structure
# Format: base_models/<model_name>_pass@<k>/
model_output_dir="${OUTPUT_DIR}/${model_clean_name}_pass${HIGH_PASS_K}"
mkdir -p ${model_output_dir}

echo "Results will be saved to: ${model_output_dir}"

# Project configuration
project_name=verl_exploration
experiment_name="${model_clean_name}_base"

# Context configuration
response_length=3072
prompt_length=1024
total_ctx=$((prompt_length + response_length))

# Validation batch size (smaller for high pass@k)
# 8 for pass@1024, 16 for pass@512
val_batch_size=8

# GPU and batch configuration
ppo_mini_batch_size=512
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=160
n_gpus=8
nnodes=1

# GSPO-specific configuration (for config compatibility)
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

# Evaluation datasets
val_files="[/fast/pmayilvahanan/datasets/aime_2024/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/aime_2025/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/math_b_exploration/qwen25math15_unsolved_no_suffix.parquet]"

# Ensure results directory exists
results_dir=results
mkdir -p ${results_dir}

# Track evaluation time
eval_start_time=$(date +%s)

echo ""
echo "================================================================================"
echo "Starting base model evaluation"
echo "================================================================================"
echo "Model: ${MODEL_NAME}"
echo "Pass@k: ${HIGH_PASS_K}"
echo "Output: ${model_output_dir}"
echo "================================================================================"
echo ""

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
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=${total_ctx} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${total_ctx} \
    actor_rollout_ref.rollout.n=${HIGH_PASS_K} \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
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
    trainer.default_local_dir=${model_output_dir} \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.val_before_train=True \
    +trainer.eval_filename=evals_base_pass${HIGH_PASS_K}.jsonl \
    reward_model.reward_manager=dapo

# Calculate evaluation time
eval_end_time=$(date +%s)
eval_duration=$((eval_end_time - eval_start_time))
eval_minutes=$((eval_duration / 60))
eval_seconds=$((eval_duration % 60))

echo ""
echo "================================================================================"
echo "Base model evaluation complete!"
echo "================================================================================"
echo "Model: ${MODEL_NAME}"
echo "Pass@k: ${HIGH_PASS_K}"
echo "Evaluation time: ${eval_minutes}m ${eval_seconds}s"
echo "-------------------------------------------------------------------------------"
echo "Results saved to:"
echo "  - ${model_output_dir}/evals_base_pass${HIGH_PASS_K}.jsonl"
echo "  - ${results_dir}/${experiment_name}_evals_base_pass${HIGH_PASS_K}.jsonl"
echo "Traces saved to:"
echo "  - ${model_output_dir}/validation_data_base_pass${HIGH_PASS_K}/"
echo "================================================================================"
echo ""
echo "To view results:"
echo "  python view_eval_results.py ${model_output_dir}/evals_base_pass${HIGH_PASS_K}.jsonl --metrics pass@${HIGH_PASS_K}"
echo ""

