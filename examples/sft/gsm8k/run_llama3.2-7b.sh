set -x
# not working sadly
# IMPORTANT: for 7B, we need to use the llama-2-7b-hf model and remove the chat template (data.chat_template=null)

experiment_name=llama3.2-7b_sft
save_path=/fast/pmayilvahanan/post_training/verl_checkpoints/$experiment_name

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/fast/pmayilvahanan/datasets/gsm8k/train.parquet \
    data.val_files=/fast/pmayilvahanan/datasets/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    +data.chat_template=null \
    data.micro_batch_size=4 \
    model.partial_pretrain=meta-llama/Llama-2-7b-hf \
    model.enable_gradient_checkpointing=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=verl_grpo_example_gsm8k \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb','local_json'] \
    trainer.total_epochs=2 \
    trainer.default_hdfs_dir=null \
    +trainer.save_checkpoint_steps=6 \
    +trainer.save_config=True \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true  $@ 