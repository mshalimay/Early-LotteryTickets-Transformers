# Short Summary 
This work proposes a novel method for early identification of Lottery-Tickets in Transformer-Based architechtures. It utilizes the nuclear norm of pruning masks as measure
of information combined with time-series techniques to assess their stationarity, resulting in more stable identification and fewer false positives than current practices involving the Hamming Distances of consecutive masks.

This repo implements:
- Structured pruning for Transformer architechtures
- Early Identification of Lottery Tickets (LT) using time-series techniques for stationarity detection in slimming coefficient trajectories
- Early LT identification and pruning during finetuning of ViT and Bert models
- Early LT identification and pruning during pre-training of a small ViT on CIFAR-10

# Directory Structure
The `Finetuning` directory contains the code to replicate results of finetuning with Bert and Google ViT model.
Steps:
- Install the `Transformers` library. 
- Substitute the selected files `Transformers` by the files in `transformers_modif_files`.
- Download the necessary checkpoints and data.
- Run `Finetuning/scripts/slim_llm.bash` and `Finetuning/scripts/slim_vit.bash` to finetune the models with network slimming and collect the slimming coefficients for the full finetuning stage.
- Run `Finetuning/scripts/prune_train.sh` to prune the models and train the pruned models at different epochs.
- Please modify parameters in the bash files according to your needs.
- The jupyter notebook `analysis_llm.ipynb` use the weight outputs of previous steps. It contains examples of implementation of the EB strategies and generation of the results in the report. 

The `ViT_pretraining` directory contains the code to replicate results for pre-training of a small ViT
**NOTE**: different from finetuning, here a small ViT model is built from scratch, so there is no need to replace `Transformers` files
Steps:
- run `ViT_pretraining/setup.sh`
- run `ViT_pretraining/scripts/slimming.sh` for pretraining with network slimming.
- run  `ViT_pretraining/scripts/prunning.sh` for pruning and training of the pruned model.
- The notebook `analysis_vit_pretrain.ipynb`  use the weight outputs of previous steps. It contains examples of implementation of the EB strategies and generation of the results in the report. 
