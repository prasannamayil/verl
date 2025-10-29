#!/bin/bash
set -x

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

module load cuda/12.9
source ~/.verl_102025/bin/activate

# Experiment configuration
dataset=dapo_5k
model_name=Qwen/Qwen2.5-Math-1.5B
rollouts=8
epochs=20
project_name=verl_exploration
n_gpus=8
nnodes=1

# Model configuration for GSPO
response_length=1024  # 8k tokens as per GSPO paper
prompt_length=3072
total_ctx=$((prompt_length + response_length))
batch_size=1024
ppo_mini_batch_size=512
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=160

# GSPO-specific configuration
adv_estimator=grpo
loss_mode=gspo
loss_agg_mode="seq-mean-token-mean"
learning_rate=1e-6

# GSPO clipping parameters
clip_ratio_low=0.0003  # as recommended by the paper
clip_ratio_high=0.0004 # as recommended by the paper

# No KL for GSPO
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

# Entropy-Based Advantage Shaping Parameters
use_entropy_advantage_shaping=true
entropy_advantage_alpha=0.4  # Alpha coefficient for entropy term
entropy_advantage_kappa=2.0  # Kappa coefficient for bounding


# Experiment naming
experiment_name=gspo_entropy_advantage_qwen25math_1.5b_${dataset}_epochs${epochs}_rollouts${rollouts}_alpha${entropy_advantage_alpha}_kappa${entropy_advantage_kappa}
checkpoint_dir=/fast/pmayilvahanan/verl_checkpoints/exploration/${experiment_name}
# Run training with GSPO + Entropy Advantage Shaping
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_entropy_advantage_shaping=${use_entropy_advantage_shaping} \
    actor_rollout_ref.actor.entropy_advantage_alpha=${entropy_advantage_alpha} \
    actor_rollout_ref.actor.entropy_advantage_kappa=${entropy_advantage_kappa} \
    data.train_files=/fast/pmayilvahanan/datasets/dapo_math_17k/dapo_non_matching_math_b_5k.parquet \
    data.val_files=[/fast/pmayilvahanan/datasets/aime_2024/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/aime_2025/test_nosuffix.parquet,/fast/pmayilvahanan/datasets/math_b_exploration/qwen25math15_unsolved_no_suffix.parquet] \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=${total_ctx} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${total_ctx} \
    actor_rollout_ref.rollout.n=${rollouts} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','local_json'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=${nnodes} \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.total_epochs=${epochs} \
    trainer.default_local_dir=${checkpoint_dir} \
    +trainer.save_config=True \
    reward_model.reward_manager=dapo