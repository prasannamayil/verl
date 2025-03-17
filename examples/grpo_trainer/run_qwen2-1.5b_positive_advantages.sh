set -x
# BEWARE OF RESPONSE LENGTH and PROMPT LENGTH
export VLLM_ATTENTION_BACKEND=XFORMERS

dataset=$1
epochs=$2
project_name=advantages_stay_positive
dirr=advantages_stay_positive

# important args
response_length=1024  # usually 1024 (for GSM8K, otherwise might wanna try twice)
prompt_length=1024  # usually 1024 (for GSM8K, otherwise might wanna try twice)
batch_size=1024  # usually 1024 (for GSM8K, otherwise might wanna try half)
ppo_mini_batch_size=256  # usually 256 (for GSM8K, otherwise might wanna try half)
ppo_micro_batch_size_per_gpu=80  # usually 80 (for GSM8K, otherwise might wanna try half)
log_prob_micro_batch_size_per_gpu=160  # usually 160 (for GSM8K, otherwise might wanna try half)


experiment_name=qwen2_1.5b_grpo_${dataset}_epochs_${epochs}_only_positive_advantages
checkpoint_dir=/fast/pmayilvahanan/post_training/verl_checkpoints/${dirr}/${experiment_name}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/fast/pmayilvahanan/datasets/${dataset}/train.parquet \
    data.val_files=/fast/pmayilvahanan/datasets/${dataset}/test.parquet \
    data.train_batch_size=${batch_size} \
    data.max_prompt_length=${prompt_length} \
    data.max_response_length=${response_length} \
    actor_rollout_ref.model.path=Qwen/Qwen2-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.n=5 \
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
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=$epochs \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.track_advantages=False \
    trainer.track_advantages_freq=7 \
    trainer.save_config=True \
    trainer.only_positive_advantages=True