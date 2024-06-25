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

overwrite_cache=$(get_flag 'overwrite-cache' false)
nocuda=$(get_flag 'nocuda' false)
doeval=$(get_flag 'do-eval' false)
fp16=$(get_flag 'fp16' false)
overwrite_output_dir=$(get_flag 'overwrite-output-dir' true)

dataset='squad'
seed=42
model_path='bert-base-uncased'
model_type='bert'

self_slimming=$(get_flag 'self-slimming' true)
inter_slimming=$(get_flag 'inter-slimming' true)

train_epochs=2
train_batch_size=12
eval_batch_size=12

learning_rate=3e-5
l1_self=1e-4
l1_inter=1e-4
train_file=./data/squad1.1/train-v1.1.json
predict_file=./data/squad1.1/dev-v1.1.json

output_dir=./squad_outputs/


slim_before=$(get_flag 'slim-before' false)
soft_by_one=$(get_flag 'soft-by-one' true)
python run_squad.py \
    --model-type $model_type \
    --model-name-or-path $model_path \
    --output-dir $output_dir \
    --overwrite-output-dir \
    --do-lower-case \
    --do-train \
    $doeval \
    $fp16 \
    --num-train-epochs $train_epochs \
    --per-gpu-train-batch-size $train_batch_size \
    --per-gpu-eval-batch-size $eval_batch_size \
    --learning-rate $learning_rate \
    --max-seq-length 384 \
    --doc-stride 128 \
    --l1-self $l1_self \
    --l1-inter $l1_inter \
    --seed $seed \
    $nocuda \
    $overwrite_cache \
    $self_slimming \
    $inter_slimming \
    $slim_before \
    $soft_by_one \
