set -x
# BEWARE OF RESPONSE LENGTH and PROMPT LENGTH
export VLLM_ATTENTION_BACKEND=XFORMERS

dataset=$1
epochs=$2
project_name=verl_active_grpo
dirr=active_grpo
lr=1e-5

# important args
response_length=2048  # usually 1024
prompt_length=2048  # usually 1024
batch_size=256  # usually 1024
ppo_mini_batch_size=128  # usually 256
ppo_micro_batch_size_per_gpu=40  # usually 80
log_prob_micro_batch_size_per_gpu=80  # usually 160
rollout_n=10


experiment_name=qwen2.5_1.5b_grpo_${dataset}_epochs_${epochs}_lr_${lr}_rollout_n_${rollout_n}
checkpoint_dir=/fast/pmayilvahanan/post_training/verl_checkpoints/${dirr}/${experiment_name}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/fast/pmayilvahanan/datasets/${dataset}/train.parquet \
    data.val_files=[/fast/pmayilvahanan/datasets/math/test.parquet,/fast/pmayilvahanan/datasets/aime_2024/test.parquet] \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.n_advantage_tracking=15 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','local_json'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=$epochs \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.track_advantages=False \
    trainer.track_advantages_freq=7 \
    trainer.save_config=True