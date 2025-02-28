set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

path_to_sft_checkpoint=/fast/pmayilvahanan/post_training/verl_checkpoints/llama3.2-1b_sft/global_step_29
experiment_name=llama3.2-1b_function_rm_sft_init_gsm8k_epoch_1
checkpoint_dir=/fast/pmayilvahanan/post_training/verl_checkpoints/$experiment_name

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/fast/pmayilvahanan/datasets/gsm8k/train.parquet \
    data.val_files=/fast/pmayilvahanan/datasets/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$path_to_sft_checkpoint \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','local_json'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.track_advantages=True \
    +trainer.track_advantages_freq=1 \
    +trainer.save_config=True $@