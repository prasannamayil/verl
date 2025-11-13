#!/bin/bash
set -x

# ============================================================================
# Hybrid GCS Approach: Save Locally + Auto-Sync to GCS
# ============================================================================
# This approach:
# 1. Saves checkpoints to fast local storage (no network overhead during training)
# 2. Automatically syncs to GCS after each checkpoint save
# 3. More reliable than gcsfuse for large checkpoint files
# 4. Can resume from GCS by syncing down first
# ============================================================================

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1

#module load cuda/12.9
source ~/.verl_102025/bin/activate

# Experiment configuration
dataset=dapo_17k_non_matching
model_name=Qwen/Qwen3-8B-Base
rollouts=8
epochs=10
project_name=verl_exploration
n_gpus=8
nnodes=1

# Model configuration for GSPO
response_length=7168  # 8k tokens as per GSPO paper
prompt_length=1024
total_ctx=$((prompt_length + response_length))
batch_size=1024  # Increased from 1024 (25% increase for better GPU utilization)
ppo_mini_batch_size=512  # Increased proportionally from 512
ppo_micro_batch_size_per_gpu=8 # Increased from 8 for better GPU utilization
log_prob_micro_batch_size_per_gpu=64  # Increased from 64

# GSPO-specific configuration
adv_estimator=grpo
loss_mode=gspo
# loss_agg_mode="seq-mean-token-mean"
loss_agg_mode="seq-mean-token-sum"
learning_rate=1e-6

# GSPO clipping parameters
# clip_ratio_low=0.0003  # Original tight clipping
# clip_ratio_high=0.0006 # Original tight clipping
clip_ratio_low=0.0003  # as recommended by the paper
clip_ratio_high=0.0004 # as recommended by the paper

# Dr. GRPO / Dr. GSPO parameters
norm_adv_by_std_in_grpo=false

# No KL for GSPO
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# Experiment naming
experiment_name=gspo_qwen3_8b_base_${dataset}_epochs${epochs}_rollouts${rollouts}_bsz${batch_size}_resp_len${response_length}_norm_adv_by_std_in_grpo${norm_adv_by_std_in_grpo}_clip_ratio_low${clip_ratio_low}_clip_ratio_high${clip_ratio_high}_loss_agg_mode${loss_agg_mode}

# Paths
local_checkpoint_dir=/root/verl_checkpoints/exploration/gspo/${experiment_name}
gcs_checkpoint_path=gs://cohere-data/prasanna_dev/verl_checkpoints/exploration/gspo/${experiment_name}

echo "Local checkpoint directory: ${local_checkpoint_dir}"
echo "GCS checkpoint path: ${gcs_checkpoint_path}"

# ============================================================================
# Resume Logic: Check if checkpoints exist in GCS and sync down if needed
# ============================================================================
echo "Checking for existing checkpoints in GCS..."
if gsutil -q stat "${gcs_checkpoint_path}/latest_checkpointed_iteration.txt" 2>/dev/null; then
    echo "Found existing checkpoints in GCS. Syncing to local..."
    mkdir -p $(dirname ${local_checkpoint_dir})
    
    # Sync from GCS to local (only if files don't exist locally or are older)
    gsutil -m rsync -r -u "${gcs_checkpoint_path}/" "${local_checkpoint_dir}/"
    
    if [ $? -eq 0 ]; then
        echo "Successfully synced checkpoints from GCS. Training will resume."
    else
        echo "WARNING: Failed to sync from GCS. Starting fresh or check your gcloud auth."
    fi
else
    echo "No existing checkpoints found in GCS. Starting fresh training."
    mkdir -p ${local_checkpoint_dir}
fi

# ============================================================================
# Background sync function - syncs to GCS every 5 minutes
# ============================================================================
sync_to_gcs() {
    while true; do
        sleep 300  # Wait 5 minutes
        if [ -d "${local_checkpoint_dir}" ]; then
            echo "[$(date)] Auto-syncing checkpoints to GCS..."
            gsutil -m rsync -r -u "${local_checkpoint_dir}/" "${gcs_checkpoint_path}/" &
        fi
    done
}

# Start background sync (uncomment if you want continuous syncing)
# sync_to_gcs &
# SYNC_PID=$!

# ============================================================================
# Run Training
# ============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    data.train_files=/root/repos/verl/datasets/dapo_math_17k/dapo_non_matching_math_b_full.parquet \
    data.val_files=[/root/repos/verl/datasets/aime_2024/test_nosuffix.parquet,/root/repos/verl/datasets/aime_2025/test_nosuffix.parquet,/root/repos/verl/datasets/math-b/qwen3_8bbase_unsolved_no_suffix.parquet] \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${model_name} \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=${total_ctx} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${total_ctx} \
    actor_rollout_ref.rollout.n=${rollouts} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','local_json'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=${epochs} \
    trainer.default_local_dir=${local_checkpoint_dir} \
    trainer.max_actor_ckpt_to_keep=2 \
    +trainer.save_config=True \
    reward_model.reward_manager=dapo

TRAIN_EXIT_CODE=$?

# Kill background sync if it was started
# [ ! -z "$SYNC_PID" ] && kill $SYNC_PID 2>/dev/null

# ============================================================================
# Final sync to GCS after training completes
# ============================================================================
echo "Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "Performing final sync to GCS..."

gsutil -m rsync -r -u "${local_checkpoint_dir}/" "${gcs_checkpoint_path}/"

if [ $? -eq 0 ]; then
    echo "✓ Successfully synced all checkpoints to GCS: ${gcs_checkpoint_path}"
    echo "✓ You can now safely delete local checkpoints to save space (or keep for faster resume)"
else
    echo "✗ WARNING: Final GCS sync failed! Checkpoints are only saved locally at: ${local_checkpoint_dir}"
    echo "  Manually sync with: gsutil -m rsync -r ${local_checkpoint_dir}/ ${gcs_checkpoint_path}/"
fi

exit $TRAIN_EXIT_CODE

