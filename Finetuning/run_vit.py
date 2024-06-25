import copy
import datetime
import json
import os
import random
import numpy as np
from transformers import ViTImageProcessor, DeiTImageProcessor
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW, AutoConfig, get_linear_schedule_with_warmup
import torch.nn as nn
import argparse
from utils import l1_loss
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers.models.vit.modeling_vit import ViTLayer, ViTSelfAttention, ViTAttention

from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor

mod_layer = ViTLayer
mod_self = ViTSelfAttention
mod_att = ViTAttention

# directories to save outputs, incl slimming coefs
time_anottate = datetime.datetime.now().strftime('%Y%m_%d_%H%M')
global output_dir 
output_dir = f'./vit_outputs/{time_anottate}/'
os.makedirs(output_dir, exist_ok=True)
torch.set_float32_matmul_precision('medium')


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="batch size for training",)        

    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size for evaluation",)

    parser.add_argument('--model-path', type=str, default='google/vit-base-patch16-224', help='model path')

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='c10',
        help="dataset to finetune",
    )
    #-----------------------------------------------------------
    # slimming parameters
    #-----------------------------------------------------------
    parser.add_argument(
        '--self-slimming',
        action='store_true',
        help='slimming for self attention',
    )

    parser.add_argument(
        '--inter-slimming',
        action='store_true',
        help='slimming for inter layers',
    )

    parser.add_argument(
        "--l1-self",
        type=float,
        default=0.0,
        help="coefficient for the l1 loss of network slimming for MSA modules",
    )

    parser.add_argument(
        "--l1-inter",
        type=float,
        default=0.0,
        help="coefficient for the l1 loss of network slimming for MLP modules",
    )

    parser.add_argument(
        "--slim-before",
        action='store_true',
        help="slim attention before softmax",
    )

    parser.add_argument(
        "--soft-by-one",
        action='store_true',
        help="use softmax off by one in to compute attention probs",
    )

    #-----------------------------------------------------------
    # pruning params
    #-----------------------------------------------------------
    parser.add_argument(
        "--self_pruning_ratio",
        type=float,
        default=0.0,
        help="self slimming pruning ratio",
    )
    parser.add_argument(
        "--inter_pruning_ratio",
        type=float,
        default=0.0,
        help="inter slimming pruning ratio",
    )
    
    parser.add_argument(
        "--self_pruning_method",
        type=str,
        default='layerwise',
        help="pruning method by default",
    )
    parser.add_argument(
        "--inter_pruning_method",
        type=str,
        default='global',
        help="pruning method by default",
    )
    parser.add_argument(
        "--prune_before_train",
        action='store_true',
        help='perform pruning before the training starts',
    )
    parser.add_argument(
        "--prune_before_eval",
        action='store_true',
        help='perform pruning before the evaluation starts',
    )
    parser.add_argument(
        "--load_from_pruned",
        action='store_true',
        help='load the model from a pruned checkpoint',
    )

    parser.add_argument(
        "--self_slimming_coef_file",
        type=str,
        default=None,
        required=False,
        help='path to the file that stores the history of slimming coefficients',
    )
    parser.add_argument(
        "--inter_slimming_coef_file",
        type=str,
        default=None,
        required=False,
        help='path to the file that stores the history of slimming coefficients',
    )
    parser.add_argument(
        "--slimming_coef_step",
        type=float,
        default=0,
        help='take which step of the slimming coefficients',
    )
    parser.add_argument(
        "--prune-and-train",
        action='store_true',
        help='false => search for tickets; true => prune and train',
    )

    parser.add_argument(
        "--model_name_or_path",
        default='google/vit-base-patch16-224-in21k',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument("--data_dir",default=None, type=str, help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",)
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)    
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument("--precision", default=32, type=int)

    parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')

    parser.add_argument('--load-from', type=str, default=None, help='path to checkpoint')
    
    args = parser.parse_args()
    return args


class ViTLightningModule(pl.LightningModule):
    def __init__(self, args, model, train_ds, config):
        super(ViTLightningModule, self).__init__()
        self.args = args
        self.config = config
        self.model = model
        self.step = 0
        if args.prune_and_train:
            pruned_heads, pruned_inter_neurons = prune(args, model, train_ds, config)
            self.log.info('{} neurons are pruned'.format(len(pruned_inter_neurons)))
            self.log.info(pruned_heads)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        train_dataloader = self.train_dataloader()
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        return [self.optimizer], [self.scheduler]

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)  

        if args.l1_self > 0:
            loss += l1_loss(mod_self, self.model, args.l1_self)

        if args.l1_inter > 0:
            loss += l1_loss(mod_layer, self.model, args.l1_inter)

        # logs metrics for each training_step, # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        if args.self_slimming or args.inter_slimming:
            record_slimming(self.model, mod_self, mod_layer)

        self.step+=1

        if self.step % save_every == 0:
            save_slimming(self.args)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        return loss

    def train_dataloader(self):
        return train_dataloader

    def test_dataloader(self):
        return test_dataloader

def save_slimming(args):
    if args.self_slimming:
        self_coefs = [np.stack(self_slimming_coef, axis=0) for self_slimming_coef in self_slimming_coef_records]        
        np.save(os.path.join(args.output_dir, 'self_slimming_coef_records.npy'), np.stack(self_coefs, axis=0))

    if args.inter_slimming:
        inter_coefs = [np.stack(inter_slimming_coef, axis=0) for inter_slimming_coef in inter_slimming_coef_records]
        np.save(os.path.join(args.output_dir, 'inter_slimming_coef_records.npy'), np.stack(inter_coefs, axis=0))

def record_slimming(model, mod_msa, mod_layer):
    # Record the learnable coefficients after each step of update
    global self_slimming_coef_records, inter_slimming_coef_records
    idx_layer = 0
    for m in model.modules():
        if isinstance(m, mod_msa) and m.slimming:
            self_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
            idx_layer += 1

    idx_layer = 0
    for m in model.modules():
        if isinstance(m, mod_layer) and m.slimming:
            inter_slimming_coef_records[idx_layer].append(m.slimming_coef.detach().cpu().numpy().reshape(-1))
            idx_layer += 1

        
        
def prune(args, model, new_config):
    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_attention_heads)
    if args.slimming_coef_step > 0:
        slimming_coefs = np.load(args.self_slimming_coef_file)
    else:
        # Random pruning
        # Ret internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-args.slimming_coef_step)
        slimming_coefs = np.random.rand(len(attention_modules), new_config.num_attention_heads)
        # Reset internal state
        np.random.set_state(rand_state)

    total_num_steps = slimming_coefs.shape[1]
    
    # Calculate which step to draw ticket
    if args.slimming_coef_step >= 1.0:
        slimming_coefs = slimming_coefs[:, int(args.slimming_coef_step), :]
    elif args.slimming_coef_step > 0.0:
        slimming_coefs = slimming_coefs[:, round(args.slimming_coef_step * total_num_steps), :]
    else:
        slimming_coefs = slimming_coefs[:, np.floor(args.slimming_coef_step), :]

    # Prune self-attention heads based on the learnable coefficients
    attention_modules = [m for m in model.modules() if isinstance(m, mod_att)]

    # If we do layerwise pruning, calculate the threshold along the last dimension
    # of `slimming_coefs`, which corresponds to the self-attention heads in each layer;
    # otherwise, calculate the threshold along all dimensions in `slimming_coefs`.
    quantile_axis = -1 if args.self_pruning_method == 'layerwise' else None
    threshold = np.quantile(slimming_coefs, args.self_pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = slimming_coefs > threshold
    for m, mask in zip(attention_modules, layers_masks):
        pruned_heads = [i for i in range(len(mask)) if mask[i] == 0]
        # logger.info(pruned_heads)
        # call transformer's method to prune the heads
        m.prune_heads(pruned_heads)

    # Prune intermediate neurons in FFN modules based on the learnable coefficients
    bert_layers = [m for m in model.modules() if isinstance(m, mod_layer)]

    # Get the coefficients for pruning, which has shape (num_hidden_layers, num_inter_neurons)
    if args.slimming_coef_step > 0:
        slimming_coefs = np.load(args.inter_slimming_coef_file)[:, args.slimming_coef_step-1, :]
    else:
        # Random pruning
        # Get internal state of the random generator first
        rand_state = np.random.get_state()
        # Set random seed
        np.random.seed(-args.slimming_coef_step + 1)
        slimming_coefs = np.random.rand(
            len(bert_layers), bert_layers[0].intermediate.dense.out_features)
        # Reset internal state
        np.random.set_state(rand_state)

    quantile_axis = -1 if args.inter_pruning_method == 'layerwise' else None
    threshold = np.quantile(slimming_coefs, args.inter_pruning_ratio, axis=quantile_axis, keepdims=True)
    layers_masks = slimming_coefs > threshold
    for m, mask in zip(bert_layers, layers_masks):
        pruned_inter_neurons = [i for i in range(new_config.intermediate_size) if mask[i] == 0]
        # logger.info('{} neurons are pruned'.format(len(pruned_inter_neurons)))
        m.prune_inter_neurons(pruned_inter_neurons)

    return pruned_heads, pruned_inter_neurons

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus > 0:
        torch.cuda.manual_seed_all(args.seed)
#===============================================================================
# main
#===============================================================================
args = parse_args()
args.output_dir = output_dir

# args.self_slimming=True
# args.inter_slimming=True
# args.l1_self=1e-4
# args.l1_inter=1e-4
# args.num_train_epochs=1
# args.dataset = 'imnet1k'

no_cuda = args.no_cuda and torch.cuda.is_available()
args.gpus = 0 if no_cuda else args.gpus

if args.precision==16:
    args.precision = '16-mixed'



set_seed(args)

# save args as json
with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f)

#==============================================================================
# load & proccess data
#==============================================================================
from utils import get_dataset
train_ds, test_ds = get_dataset(args)



# data processing and augmentations
if 'deit' in args.model_path.lower():
    processor = DeiTImageProcessor.from_pretrained(args.model_path)
elif 'google' in args.model_path.lower():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
else:
    raise NotImplementedError   

image_mean = processor.image_mean
image_std = processor.image_std
size = processor.size["height"]
normalize = Normalize(mean=image_mean, std=image_std)

def get_key(examples, possible_keys=['img', 'image']):
    for key in possible_keys:
        if key in examples:
            return key
    raise ValueError('No key found in the dataset')

def train_transforms(examples):
    img_key = get_key(examples)
    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(),normalize,])
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples[img_key]]
    return examples

def val_transforms(examples):
    img_key = get_key(examples)
    _val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize,])
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples[img_key]]
    return examples

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=8)
test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=args.eval_batch_size, num_workers=8)

#==============================================================================
# load and process the model
#==============================================================================
# DEBUG
# args.model_path = 'facebook/deit-small-patch16-224'
# args.self_slimming=True
# args.inter_slimming=True


config = AutoConfig.from_pretrained(args.model_path)

if not args.prune_and_train:
    config.self_slimming = args.self_slimming
    config.inter_slimming = args.inter_slimming
    global self_slimming_coef_records, inter_slimming_coef_records
    self_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]
    inter_slimming_coef_records = [[] for _ in range(config.num_hidden_layers)]

new_config = config
if args.prune_and_train and args.load_from_pruned:
    new_config = copy.deepcopy(config)
    new_config.self_pruning_ratio = args.self_pruning_ratio
    new_config.inter_pruning_ratio = args.inter_pruning_ratio

if args.dataset=='c10':
    new_config.num_labels=10
elif args.dataset=='c100' or args.dataset=='imnet100':
    new_config.num_labels=100
elif args.dataset=='imnet200':
    new_config.num_labels=200
elif args.dataset=='imnet1k':
    print("Using Imagenet 1k")
    new_config.num_labels=1000
else:
    raise NotImplementedError


new_config.slim_before = args.slim_before
new_config.off_by_one = args.soft_by_one


if args.load_from:
    vit = ViTForImageClassification.from_pretrained(args.load_from, config=new_config, ignore_mismatched_sizes=True)
else:
    vit = ViTForImageClassification.from_pretrained(args.model_path, config=new_config, ignore_mismatched_sizes=True)

# torch.save(vit.state_dict(), os.path.join(args.output_dir, 'initial_model.pth'))
logger = pl.loggers.CSVLogger(save_dir=args.output_dir,)
refresh_rate = 1

model = ViTLightningModule(args, vit, train_ds, new_config)

# Configure the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',  
    dirpath=args.output_dir,    # specify where you want to save the model
    filename='vit-{epoch:02d}-{validation_loss:.2f}',
    save_top_k=1,               # the number of best models to save; saves only the top 3 models
    mode='min',                 # 'min' to save models with minimal 'validation_loss'
    save_last=True              # also save the last model at the end of training
)

# frequency to save slimming coefficients
save_every=len(train_dataloader)//2

if args.no_cuda:
    trainer = pl.Trainer(precision=args.precision, accelerator="cpu", logger=logger,  max_epochs=int(args.num_train_epochs),
                         enable_model_summary=True, callbacks=[checkpoint_callback])
else:
    trainer = pl.Trainer(precision=args.precision, devices=args.gpus, accelerator='gpu', logger=logger, 
                         enable_model_summary=True, max_epochs=int(args.num_train_epochs), callbacks=[checkpoint_callback])

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

if not args.prune_and_train:
    save_slimming(args)

