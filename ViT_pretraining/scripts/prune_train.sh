#!bin/bash


slimmig_coefs_msa_file='./slimming_coefs/vit_c10_aa_ls_202404220250/msa_slimming_coefs.npy'
slimmig_coefs_mlp_file='./slimming_coefs/vit_c10_aa_ls_202404220250/mlp_slimming_coefs.npy'

lr=1e-4
prune_step=10

python main.py --dataset c10 --label-smoothing --autoaugment \
        --mlp-prune-ratio .4 \
        --msa-prune-ratio .4 \
        --prune-step $prune_step \
        --msa-slimming-coefs-file $slimmig_coefs_msa_file \
        --mlp-slimming-coefs-file $slimmig_coefs_mlp_file \
        --lr $lr

