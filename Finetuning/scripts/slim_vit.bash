#!/bin/bash
get_flag() {
    local flag=$1
    local condition=$2
    if [ "$condition" = true ]; then
        echo "--$flag"
    else
        echo ""
    fi
}


model_path='google/vit-base-patch16-224'   # default
# model_path='facebook/deit-tiny-patch16-224'
# model_path=facebook/deit-small-patch16-224 
# model_path=facebook/deit-base-patch16-224 

load_from=''

train_epochs=3  
train_batch_size=64
eval_batch_size=64

self_slimming=$(get_flag 'self-slimming' true)
inter_slimming=$(get_flag 'inter-slimming' true)

l1_self=1e-4
l1_inter=1e-4
learning_rate=3e-5
dataset='imnet1k'
precision=16

slim_before=$(get_flag 'slim-before' false)
soft_by_one=$(get_flag 'soft-by-one' false)
python ./ViT_pretraining/run_vit.py \
    --model-path $model_path \
    --load-from $load_from \
    --dataset $dataset  \
    --l1-self $l1_self \
    --l1-inter $l1_inter \
    --learning_rate $learning_rate \
    --precision $precision \
    $self_slimming \
    $inter_slimming \
    $slim_before \
    $soft_by_one \
    --num_train_epochs $train_epochs \
    --train_batch_size $train_batch_size \
    --eval_batch_size $eval_batch_size \
