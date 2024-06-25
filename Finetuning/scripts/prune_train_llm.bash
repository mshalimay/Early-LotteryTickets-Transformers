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
model_type='bert'

train_epochs=2
train_batch_size=12
eval_batch_size=12

learning_rate=3e-5
train_file=./data/squad1.1/train-v1.1.json
predict_file=./data/squad1.1/dev-v1.1.json

output_dir=./squad_outputs/

model_dir='./squad_outputs/202404_26_2131'
output_dir="${model_dir}/pruned"

self_pruning_method="layerwise"
inter_pruning_method="global"


#-------------------------------------------------------------------------------
self_pruning_ratio=0.8
inter_pruning_ratio=0.8
slimming_coef_steps=(2955 8864 11819)

for slimming_coef_step in ${slimming_coef_steps[@]}; do
    python run_squad.py \
        --model-type $model_type \
        --model-name-or-path "${model_dir}/checkpoint-0" \
        --do-train \
        $doeval \
        --do-lower-case \
        $fp16 \
        --num-train-epochs $train_epochs \
        --per-gpu-train-batch-size $train_batch_size \
        --per-gpu-eval-batch-size $eval_batch_size \
        --learning-rate $learning_rate \
        --max-seq-length 384 \
        --doc-stride 128 \
        --seed $seed \
        $nocuda \
        $overwrite_cache \
        --prune-and-train \
        --self-slimming-coef-file  "${model_dir}/self_slimming_coef_records.npy"  \
        --inter-slimming-coef-file "${model_dir}/inter_slimming_coef_records.npy" \
        --overwrite-output-dir \
        --output-dir $output_dir \
        --self-pruning-ratio $self_pruning_ratio \
        --self-pruning-method $self_pruning_method \
        --inter-pruning-ratio $inter_pruning_ratio \
        --inter-pruning-method $inter_pruning_method \
        --slimming-coef-step $slimming_coef_step \
        --do-eval \
        --save-steps 14500 
done

