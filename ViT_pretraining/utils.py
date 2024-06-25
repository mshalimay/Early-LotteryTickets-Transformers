import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from EB_Transformers_TimeSeries.ViT_pretraining.data_processing.autoaugment import CIFAR10Policy, SVHNPolicy
from EB_Transformers_TimeSeries.ViT_pretraining.models.criterions import LabelSmoothingCrossEntropyLoss
from EB_Transformers_TimeSeries.ViT_pretraining.data_processing.da import RandomCropPaste

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

    
def get_slimming_masks(args):
    if args.mlp_prune_ratio == 0:
        mlp_masks = None
    else:
        quantile_axis = -1 if args.mlp_pruning_method == 'layerwise' else None
        mlp_coefs = np.load(args.mlp_slimming_coefs_file)[:,args.prune_step,:]
        threshold = np.quantile(mlp_coefs, args.mlp_prune_ratio, axis=quantile_axis, keepdims=True)
        mlp_masks = mlp_coefs > threshold

    if args.msa_prune_ratio == 0:
        msa_masks = None
    else:
        quantile_axis = -1 if args.msa_pruning_method == 'layerwise' else None
        msa_coefs = np.load(args.msa_slimming_coefs_file)[:,args.prune_step,:]
        threshold = np.quantile(msa_coefs, args.msa_prune_ratio, axis=quantile_axis, keepdims=True)
        msa_masks = msa_coefs > threshold

    return mlp_masks, msa_masks

    
def get_model(args):
    if args.model_name == 'vit':
        from EB_Transformers_TimeSeries.ViT_pretraining.models.vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            slim_mlp=args.slim_mlp,
            slim_msa=args.slim_msa,
            mlp_prune_ratio=args.mlp_prune_ratio,
            msa_prune_ratio=args.msa_prune_ratio,
            slim_before=args.slim_before,
            soft_by_one=args.soft_by_one
        )

        if args.prune_step > 0:
            mlp_masks, msa_masks = get_slimming_masks(args)
            if mlp_masks:
                print(f"Pruning {mlp_masks.sum()} neurons in MLP layers")
                net.slim_mlp(mlp_masks)
            if msa_masks:
                print(f"Pruning {msa_masks.sum()} heads in MSA layers")
                net.slim_msa(msa_masks)
                
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")
    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)