#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'
dataset_path=$1
train_model_path=$2
save_model_path=$3
batch_size=$4
max_length=$5

export N_GPUS=8
export WANDB_API_KEY='xxx'
# export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ray stop --force && ray start --head --include-dashboard=True

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$dataset_path"/train.parquet \
    data.val_files="$dataset_path"/test.parquet \
    data.prompt_key=input \
    data.max_length=$max_length \
    data.response_key=output \
    data.train_batch_size=$batch_size \
    data.micro_batch_size=$N_GPUS \
    model.partial_pretrain="$train_model_path" \
    trainer.default_hdfs_dir="$save_model_path" \
    trainer.logger=['console','wandb'] \
    trainer.project_name=sft \
    trainer.experiment_name="SFT_experiment" \
    trainer.total_epochs=3 \
    optim.lr=1e-6 \
    optim.weight_decay=0.01
