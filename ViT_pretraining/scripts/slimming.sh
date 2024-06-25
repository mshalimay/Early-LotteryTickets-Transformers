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

lr=1e-3         # default: 1e-3
min_lr=1e-5     # default: 1e-5
l1_mlp=1e-4     # default: 0.0
l1_msa=1e-4     # default: 0.0
patch=16        # default: 8
head=12         # num of attention heads.                  default: 12
num_layers=7    # num of encoder layers.                   default: 7
mlp_hidden=384  # num of intermediate layers in MLP block. default: 384

cutmix=$(get_flag 'cutmix' false)
mixup=$(get_flag 'mixup' false)
label_smoothing=$(get_flag 'label-smoothing' true) #True = LabelSmoothingCrossEntropyLoss
autoaugment=$(get_flag 'autoaugment' true)
slim_before=$(get_flag 'slim-before' false)
soft_by_one=$(get_flag 'soft-by-one' false)

batch_size=128
eval_batch_size=256
max_epochs=120
precision=16
#--------------------------------------------------------
# python main.py --dataset c10 $cutmix $mixup $label_smoothing $autoaugment \
#         --batch-size $batch_size \
#         --eval-batch-size $eval_batch_size \
#         --max-epochs $max_epochs \
#         --slim-mlp \
#         --slim-msa \
#         --l1-mlp $l1_mlp \
#         --l1-msa $l1_msa \
#         --lr $lr  \
#         --min-lr $min_lr \
#         --patch $patch \
#         --head $head \
#         --num-layers $num_layers \
#         --mlp-hidden $mlp_hidden \
#         --slim-before $slim_before \
#         --soft-by-one $soft_by_one \

#--------------------------------------------------------
slim_before=$(get_flag 'slim-before' true)
soft_by_one=$(get_flag 'soft-by-one' true)
python main.py --dataset c10 $cutmix $mixup $label_smoothing $autoaugment \
        --batch-size $batch_size \
        --eval-batch-size $eval_batch_size \
        --max-epochs $max_epochs \
        --slim-mlp \
        --slim-msa \
        --l1-mlp $l1_mlp \
        --l1-msa $l1_msa \
        --lr $lr  \
        --min-lr $min_lr \
        --patch $patch \
        --head $head \
        --num-layers $num_layers \
        --mlp-hidden $mlp_hidden \
        --slim-before $slim_before \
        --soft-by-one $soft_by_one \

#--------------------------------------------------------
slim_before=$(get_flag 'slim-before' true)
soft_by_one=$(get_flag 'soft-by-one' false)
python main.py --dataset c10 $cutmix $mixup $label_smoothing $autoaugment \
        --batch-size $batch_size \
        --eval-batch-size $eval_batch_size \
        --max-epochs $max_epochs \
        --slim-mlp \
        --slim-msa \
        --l1-mlp $l1_mlp \
        --l1-msa $l1_msa \
        --lr $lr  \
        --min-lr $min_lr \
        --patch $patch \
        --head $head \
        --num-layers $num_layers \
        --mlp-hidden $mlp_hidden \
        --slim-before $slim_before \
        --soft-by-one $soft_by_one \

#--------------------------------------------------------
slim_before=$(get_flag 'slim-before' false)
soft_by_one=$(get_flag 'soft-by-one' true)
python main.py --dataset c10 $cutmix $mixup $label_smoothing $autoaugment \
        --batch-size $batch_size \
        --eval-batch-size $eval_batch_size \
        --max-epochs $max_epochs \
        --slim-mlp \
        --slim-msa \
        --l1-mlp $l1_mlp \
        --l1-msa $l1_msa \
        --lr $lr  \
        --min-lr $min_lr \
        --patch $patch \
        --head $head \
        --num-layers $num_layers \
        --mlp-hidden $mlp_hidden \
        --slim-before $slim_before \
        --soft-by-one $soft_by_one \
