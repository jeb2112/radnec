# quick script for loading checkpoint results 
# for sam with BraTS2024

import torch
import numpy as np
import monai
import glob
import os
import re
import matplotlib
import matplotlib.pyplot as plt

from transformers import SamModel
from torch.optim import Optimizer
# huggingface accelerate might work on multiple AWS instances?
# otherwise replace
from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from torch.optim.lr_scheduler import LinearLR
from torch.optim import AdamW

import monai
import torchvision.ops as ops
import torch.nn.functional as F

from transformers import SamProcessor  
from torch.utils.data import DataLoader


datadir = "C:\\Users\\Chris Heyn Lab\\data\\brats2024\\raw"
validation_datadir = os.path.join(datadir,'validation_slice')
training_datadir = os.path.join(datadir,'training_slice')


class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"
    
class TrainHistory():
    def __init__(self, history=None):
        self._history = history or []

    def start_session(
        self, 
        dataset_name: str,
        optimizer_name: str,
        loss_func: str,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        prompt_args: dict,
    ):
        session = {
            "epochs": 0,
            "dataset": dataset_name, 
            "optimizer": optimizer_name,
            "loss_func": loss_func,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "prompt_args": prompt_args,
        }
        self._history.append(session)

    def update_session(self, epochs_elapsed: int):
        prev_elapsed = self._history[-1]["epochs"]
        if epochs_elapsed < prev_elapsed:
            print(f"Warning: Updated session to lower epoch value! previous={prev_elapsed}, requested={epochs_elapsed}")
        self._history[-1]["epochs"] = epochs_elapsed

    def get_history_list(self):
        return self._history


def get_model_name(model_size='base'):
    if not (model_size == "base" or model_size == "huge"):
        raise ValueError(f'Unknown model size "{model_size}".')

    if True:
        return f"facebook/sam-vit-{model_size}"
    else:
        return "C:\\Users\\Chris Heyn Lab\\data\\sam_models\\sam_vit_b_01ec64.pth"


def upsample_mask(pred_mask, target_size):
    # TODO: Should we use nearest? Or should we interpolate somehow?
    return F.interpolate(pred_mask.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)

def focal_loss_fn(pred_mask, ground_truth):
    return ops.sigmoid_focal_loss(pred_mask, ground_truth, reduction="mean")

dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
def dice_loss_fn(pred_mask, ground_truth):
    return dice_loss(pred_mask, ground_truth)

def get_linear_comb_loss(focal_loss_ratio):
    dice_loss_ratio = 1 - focal_loss_ratio

    def lc_loss(pred_mask, ground_truth):
        dice_loss = dice_loss_fn(pred_mask, ground_truth)
        focal_loss = focal_loss_fn(pred_mask, ground_truth)
        combined_loss = (focal_loss_ratio * focal_loss) + (dice_loss_ratio * dice_loss)
        return combined_loss

    return lc_loss

sam_lc_loss = get_linear_comb_loss(19/20)

class LossFunctionsConfig():
    def __init__(self, funcs, track_iou=False):
        self.names = [item[0] for item in funcs]
        self.funcs = [item[1] for item in funcs]
        self.track_iou = track_iou

default_loss_config = LossFunctionsConfig([("dice", dice_loss_fn)])


def get_optimizer(optimizer_name: str, optimizer_args: dict, model: SamModel):
    optimizer = None
    
    if optimizer_name == "AdamW":
        optimizer = AdamW(model.mask_decoder.parameters(), **optimizer_args)

    # TODO: Add more optimizer options here!

    if optimizer == None: raise NameError(f"Unrecognized optimizer '{optimizer_name}', could not initialize!")
    return optimizer

def get_prompt_descriptor_str(args: dict):
    descriptor = f"{args['prompt_type']}_"
    
    if args["prompt_type"] == PromptType.CONTROL_POINTS:
        descriptor = descriptor + f"{args['num_positive']}p{args['num_negative']}n"
        if (args["erode"]): descriptor = descriptor + "_mm=" + args["multi_mask"]
        if (args["erode"]): descriptor = descriptor + "_erode"

    if args["prompt_type"] == PromptType.BOUNDING_BOX:
        descriptor = descriptor + f"{args['perturbation']}"

    return descriptor

def get_model_descriptor_str(
    model_size,
    optimizer_name,
    learning_rate,
    weight_decay,
    batch_size,
    prompt_args,
    mixed_precision,
    loss_func,
):
    train_descriptor = f"{model_size}_{optimizer_name}"
    hyperparams_descriptor = f"lr={learning_rate}_wd={weight_decay}_bs={batch_size}_mp={mixed_precision}"
    prompt_descriptor = get_prompt_descriptor_str(prompt_args)
    
    return train_descriptor + "_" + hyperparams_descriptor + "_" + prompt_descriptor + f"_loss={loss_func}"


def load_model_checkpoint(accelerator, checkpoint_path, model: SamModel, optimizer: Optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["lr_scheduler_dict"])

    other = {
        "epoch": checkpoint["epoch"] + 1,  # Start from the next epoch
        "best_loss": checkpoint["best_loss"],
        "train_losses": checkpoint["train_losses"],
        "validation_losses": checkpoint["validation_losses"],
        "train_history": TrainHistory(checkpoint["train_history"]),
    }

    accelerator.print(f"Model checkpoint loaded from: {checkpoint_path}")
    accelerator.print(f"Resuming training from epoch: {other['epoch']}")

    return model, optimizer, scheduler, other


dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
def dice_loss_fn(pred_mask, ground_truth):
    return dice_loss(pred_mask, ground_truth)

class LossFunctionsConfig():
    def __init__(self, funcs, track_iou=False):
        self.names = [item[0] for item in funcs]
        self.funcs = [item[1] for item in funcs]
        self.track_iou = track_iou
default_loss_config = LossFunctionsConfig([("dice", dice_loss_fn)])

    
def main(
        load_checkpoint,
        loss_config:LossFunctionsConfig=default_loss_config,
        prompt_args:dict={},
        dataset_name='v1' # arbitrary tag for now
    ):
    global model_descriptor, best_checkpoint_path, train_losses, validation_losses, iou_scores

    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Instantiate the model here so that the seed also controls new weight initaliziations.
    model_name = get_model_name(model_size)
    model = SamModel.from_pretrained(model_name)

    # Optimizer and learning rate schedular instantiation.
    optimizer = get_optimizer(optimizer_name, {"lr": learning_rate, "weight_decay": weight_decay}, model)
    accelerator.print(f"Loaded {model_name} model, using {optimizer_name} optimizer")
    lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10) # TODO: Add params for this to function header?

    # Train only the decoder.
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Variables which needed to be loaded from checkpoint.
    epoch = 0
    best_loss = float("inf")
    train_losses = []
    validation_losses = []
    train_history = TrainHistory()


    if load_checkpoint is not None:
        model, optimizer, lr_scheduler, other = load_model_checkpoint(accelerator, load_checkpoint, model, optimizer, lr_scheduler)
        best_loss = other["best_loss"]
        train_losses = other["train_losses"]
        validation_losses = other["validation_losses"]
        train_history = other["train_history"]
        epoch = other["epoch"]
    a=1

# All provided loss functions will be measured. Model is
# only trained with backprop of the first loss function.
loss_config = LossFunctionsConfig([
    ("dice", dice_loss_fn),
    # ("19-1", get_linear_comb_loss(19 / 20)),  # Ratio found to be the best by the SAM authors
    # ("focal", focal_loss_fn),
], track_iou=True) # TODO: Ths currently doesn't do anything.

prompt_args = {
    # Control points version:
    # "prompt_type": PromptType.CONTROL_POINTS,
    # "num_positive": 5,
    # "num_negative": 0,
    # "erode": True,
    # "multi_mask": "mean",

    # Bounding boxes version:
    "prompt_type": PromptType.BOUNDING_BOX,
    "perturbation": 0,
    "multi_mask": None
}

args = (
        loss_config, # loss_config
        7e-6, # learning_rate
        2e-4, # weight_decay
        5, # batch_size (batch size per process is batch_size / num_processes)
        prompt_args,  # prompt_args
        42,  # seed
        None, # load_checkpoint (string path to checkpoint or None)
        None, # model_path_name (override model path or None)
)
kwargs = {
        'model_size':"base",
        'optimizer_name':"AdamW",
        'num_epochs':50,
        'mixed_precision':"fp16",
        'dataset_name':'v1', # arbitrary tag for now
        'num_processes':1
}


# main entry
    # Get model descriptor and save path.
model_size = 'base'
optimizer_name = 'AdamW'
learning_rate = 7e-6
weight_decay = 2e-4
batch_size = 5
mixed_precision = 'fp16'
model_descriptor = get_model_descriptor_str(
    model_size,
    optimizer_name,
    learning_rate,
    weight_decay,
    batch_size,
    prompt_args,
    mixed_precision,
    loss_config.names[0],
)
model_path_name = None

path_name = model_path_name or model_descriptor 
best_checkpoint_path = os.path.join(os.path.expanduser('~'),'data','sam_models','checkpoints','best _' + path_name + ".pth")
last_checkpoint_path = os.path.join(os.path.expanduser('~'),'data','sam_models','checkpoints','last_' + path_name + ".pth")

main(
        best_checkpoint_path,
        loss_config,
        prompt_args,
        dataset_name='v1'
    )
