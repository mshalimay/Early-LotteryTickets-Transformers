# Short Summary 
This work proposes a novel method for identification of Lottery-Tickets in Transformer-Based architechtures [1] early stages of training [2]. It utilizes the nuclear norm of pruning masks as measure
of information combined with time-series techniques to assess their stationarity, resulting in more stable identification and fewer false positives than current practices utilizing sequential evaluation of Hamming Distances of pruning masks.

This repo implements:
- Structured pruning for Transformer architechtures tuning network slimming during training as proposed in [6] and [16]
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


# References
[1] J. Frankle and M. Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635, 2018. Accessed: 2024-05-01.
[2] H. You, C. Li, P. Xu, Y. Fu, Y. Wang, X. Chen, R.G. Baraniuk, Z. Wang, and Y. Lin. Drawing early-bird tickets: Towards more efficient training of deep networks. arXiv preprint arXiv:1909.11957, 2019.
[3] M. Behnke and K. Heafield. Losing heads in the lottery: Pruning transformer. In The 2020 Conference on Empirical Methods in Natural Language Processing, pages 2664–2674. Association for Computational Linguistics (ACL), November 2020.
[4] S. Prasanna, A. Rogers, and A. Rumshisky. When bert plays the lottery, all tickets are winning. arXiv preprint arXiv:2005.00561, 2020.
[5] T. Chen, J. Frankle, S. Chang, S. Liu, Y. Zhang, Z. Wang, and M. Carbin. The lottery ticket hypothesis for pre-trained bert networks. Advances in neural information processing systems, 33.
[6] X. Chen, Y. Cheng, S. Wang, Z. Gan, Z. Wang, and J. Liu. Earlybert: Efficient bert training via early-bird lottery tickets. arXiv preprint arXiv:2101.00063, 2020. Accessed: 2024-05-01.
[7] V. Sze, Y. H. Chen, T. J. Yang, and J. S. Emer. Efficient processing of deep neural networks: A tutorial and survey. Proceedings of the IEEE, 105(12):2295–2329, 2017.
[8] T. Liang, J. Glossner, L. Wang, S. Shi, and X. Zhang. Pruning and quantization for deep neural network acceleration: A survey. Neurocomputing, 461:370–403, 2021.
[9] Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell. Rethinking the value of network pruning. arXiv preprint arXiv:1810.05270, October 2018. Accessed: 2024-05-01.
[10] T. Gale, E. Elsen, and S. Hooker. The state of sparsity in deep neural networks. arXiv preprint arXiv:1902.09574, 2019. Accessed: 2024-05-01.
[11] S. Anagnostidis, D. Pavllo, L. Biggio, L. Noci, A. Lucchi, and T. Hofmann. Dynamic context pruning for efficient and interpretable autoregressive transformers. Advances in Neural Information Processing Systems, 36, 2024.
[12] F. Lagunas, E. Charlaix, V. Sanh, and AM Rush. Block pruning for faster transformers. 2021.
[13] W. Kwon, S. Kim, MW Mahoney, J Hassoun, K Keutzer, and A Gholami. A fast post-training pruning framework for transformers. In Advances in Neural Information Processing Systems, volume 35, pages 24101–
24116, 2022.
[14] S. Kim, S. Shen, D. Thorsley, A. Gholami, W. Kwon, J. Hassoun, and K. Keutzer. Learned token pruning for transformers. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data
Mining, pages 784–794, 2022.
[15] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022.
[16] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang. Learning efficient convolutional networks through network slimming. In Proceedings of the IEEE international conference on computer Vision, pages 2736–2744, 2017.
[17] David N DeJong, John C Nankervis, N Eugene Savin, and Charles H Whiteman. Integration versus trend stationary in time series. Econometrica: Journal of the Econometric Society, pages 423–433, 1992.
[18] Maurice George Kendall. Rank correlation methods. 1948.
[19] Henry B Mann. Nonparametric tests against trend. Econometrica: Journal of the econometric society, pages 245–259, 1945.
[20] Richard O Gilbert. Statistical methods for environmental pollution monitoring. John Wiley & Sons, 1987.
[21] David A Dickey and Wayne A Fuller. Distribution of the estimators for autoregressive time series with a unit root. Journal of the American statistical association, 74(366a):427–431, 1979.
[22] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021
