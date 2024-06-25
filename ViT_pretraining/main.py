import argparse
import datetime
import os
import json
import torch
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
import numpy as np
from models.layers import MultiHeadSelfAttention, TransformerMLP

from utils import get_model, get_dataset, get_experiment_name, get_criterion, set_seed  
from EB_Transformers_TimeSeries.ViT_pretraining.data_processing.da import CutMix, MixUp

# TODO: make optimizer consistent with other experiment
# TODO: global pruning option


parser = argparse.ArgumentParser()
parser.add_argument("--api-key", help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="vit", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=16, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")

# slimming parameters
parser.add_argument("--slim-mlp", action="store_true", default=False)
parser.add_argument("--slim-msa", action="store_true", default=False)
parser.add_argument("--l1-mlp", default = 0.0, type=float)
parser.add_argument("--l1-msa", default = 0.0, type=float)
parser.add_argument("--save-every", default=1000, type=int)
parser.add_argument("--slim-before", action="store_true", default=False)
parser.add_argument("--soft-by-one", action="store_true", default=False)

# pruning parameters
parser.add_argument("--msa-pruning-method", default='layerwise', type=str)
parser.add_argument("--mlp-pruning-method", default='global', type=str)
parser.add_argument("--msa-prune-ratio", default=0.0, type=float)
parser.add_argument("--mlp-prune-ratio", default=0.0, type=float)
parser.add_argument("--prune-step", default=0, type=int)
parser.add_argument("--msa-slimming-coefs-file", default=None, type=str)
parser.add_argument("--mlp-slimming-coefs-file", default=None, type=str)

# parse arguments
args = parser.parse_args()

# set num_gpus and num_workers
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

set_seed(args)
args.benchmark = True if not args.off_benchmark else False


# warning for mlp_hidden
if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")


# get dataset and dataloader
train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

# pytorch lightning model
class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix: self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup: self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None
        self.step = 0

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=self.hparams.warmup_epoch, after_scheduler=self.base_scheduler)
        return [self.optimizer], [self.scheduler]

    def l1_loss(self, l1_param, mod):
        l1_loss = 0
        for m in self.model.modules():
            if isinstance(m, mod):
                l1_loss += m.slimming_coef.abs().sum()
        return l1_param*l1_loss
    
    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_= self.cutmix((img, label))
            elif self.hparams.mixup:
                if np.random.rand() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
            out = self.model(img)
            loss = self.criterion(out, label)*lambda_ + self.criterion(out, rand_label)*(1.-lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        if self.hparams.l1_mlp > 0:
            # print("\n\n\nLOSS mlp added")
            loss += self.l1_loss(self.hparams.l1_mlp, TransformerMLP)

        if self.hparams.l1_msa > 0:
            # print("\n\n\nLOSS msa added")
            loss += self.l1_loss(self.hparams.l1_msa, MultiHeadSelfAttention)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        
        if self.hparams.slim_mlp or self.hparams.slim_msa:
            self.record_slimming_coefs()

        self.step += 1
        return loss

    def record_slimming_coefs(self):
        # record slimming coefs in global variables
        idx_layer_msa, idx_layer_mlp = (0, 0)
        for m in self.model.modules():
            if isinstance(m, MultiHeadSelfAttention):
                msa_slimming_coefs[idx_layer_msa].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                idx_layer_msa += 1
            elif isinstance(m, TransformerMLP):
                mlp_slimming_coefs[idx_layer_mlp].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
                idx_layer_mlp += 1    
        # save to disk
        if self.step % args.save_every == 0:
            save_slimming()
        return
        
    def training_epoch_end(self, outputs):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")
        return


if __name__ == "__main__":
    # create experiment name
    time_anotate = datetime.datetime.now().strftime('%Y%m_%d_%H%M')
    experiment_name = get_experiment_name(args)
    experiment_name = f"{experiment_name}_{time_anotate}"
    print(experiment_name)

    # directories to save slimming coefs
    global output_dir 
    output_dir = f'./slimming_coefs/{experiment_name}_{time_anotate}/'
    os.makedirs(output_dir, exist_ok=True)

    # lists to store slimming coefficients
    global mlp_slimming_coefs, msa_slimming_coefs
    mlp_slimming_coefs = [[] for _ in range(args.num_layers)]
    msa_slimming_coefs = [[] for _ in range(args.num_layers)]

    # function to save slimming coefficients to disk
    def save_slimming():
        msa_slimming_coef = [np.stack(coefs, axis=0) for coefs in msa_slimming_coefs]
        mlp_slimming_coef = [np.stack(coefs, axis=0) for coefs in mlp_slimming_coefs]
        np.save(os.path.join(output_dir, 'msa_slimming_coefs.npy'), np.stack(msa_slimming_coef, axis=0))
        np.save(os.path.join(output_dir, 'mlp_slimming_coefs.npy'), np.stack(mlp_slimming_coef, axis=0))

    # if slimming, set prune ratios to zero to make sure
    args.mlp_prune_ratio = 0.0 if args.slim_mlp else args.mlp_prune_ratio
    args.msa_prune_ratio = 0.0 if args.slim_msa else args.msa_prune_ratio
        
    if args.api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(api_key=args.api_key,save_dir="logs",
                project_name=args.project_name,experiment_name=experiment_name)
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs",name=experiment_name)
        refresh_rate = 1

    # create pytorch lightning model
    net = Net(args)

    # save initial model params
    torch.save(net.model.state_dict(), os.path.join(output_dir, 'initial_model.pth'))

    # save args as json
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # instantiate trainer and train
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, 
                         benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, 
                         weights_summary="full", progress_bar_refresh_rate=refresh_rate,
                         val_check_interval=.4, num_sanity_val_steps=0)
    
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)


    if not args.dry_run:
        # save final slimming coefs
        save_slimming()
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)

