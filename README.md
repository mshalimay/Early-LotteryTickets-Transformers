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
