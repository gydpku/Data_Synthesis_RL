#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'
dataset_path=$1
train_model_path=$2
save_model_path=$3
temperature=$4
rollout=$5
batch_size=$6
response_length=$7
export N_GPUS=4
export WANDB_API_KEY=6ed283938a8d9f6896f0145553a1cbdaf482482e
#export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=1,2,3,4

ray stop --force && ray start --head --include-dashboard=True

export BASE_MODEL="$train_model_path"
export DATA_DIR="$dataset_path"
export ROLLOUT_TP_SIZE=4
export EXPERIMENT_NAME="$save_model_path"
export VLLM_ATTENTION_BACKEND=XFORMERS

bash TinyZero/scripts/train_tiny_zero_a100_grpo.sh $temperature $rollout $batch_size $response_length
#bash ./scripts/train_tiny_zero.sh
