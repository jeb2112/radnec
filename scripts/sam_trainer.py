# replace huggingface code here to load data
# or generator
if False:
    from datasets import load_dataset, load_from_disk, Dataset

    hf_dataset_name = "stodoran/elwha-segmentation-v1"  
    training_data = load_dataset(hf_dataset_name, split="train")  
    validation_data = load_dataset(hf_dataset_name, split="validation")


from torch.utils.data import Dataset
import torch
import numpy as np
import skimage
import monai

class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"

class SAMDataset(Dataset):  
    def __init__(  
        self,   
        dataset,   
        processor,   
        prompt_type = PromptType.BOUNDING_BOX,  
        num_positive = 3,  
        num_negative = 0,  
        erode = True,  
        multi_mask = "mean",  
        perturbation = 10,  
        image_size = (1024, 1024),  
        mask_size = (256, 256),  
    ):  
        # Asign all values to self  
        self.dataset = dataset
        self.processor = processor
        self.prompt_type = prompt_type
        self.image_size = image_size
        self.mask_size = mask_size

        # These should only be used for CONTROL_POINTS prompts.
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.erode = erode
        self.multi_mask = multi_mask

        # This is only to be used for BOUNDING_BOX prompts.
        self.perturbation = perturbation

    def __len__(self):  
        return len(self.dataset)

    def __getitem__(self, idx):  
        datapoint = self.dataset[idx]  
        input_image = skimage.resize(np.array(datapoint["image"]), self.image_size)  
        ground_truth_mask = skimage.resize(np.array(datapoint["label"]), self.mask_size)

        if self.prompt_type == PromptType.CONTROL_POINTS:  
            inputs = self._getitem_ctrlpts(input_image, ground_truth_mask)  
        elif self.prompt_type == PromptType.BOUNDING_BOX:  
            inputs = self._getitem_bbox(input_image, ground_truth_mask)

        inputs["ground_truth_mask"] = ground_truth_mask  
        return inputs
    
    # won't do control points to begin with
    if False: 
        def _getitem_ctrlpts(self, input_image, ground_truth_mask):  
            # Get control points prompt. See the GitHub for the source  
            # of this function, or replace with your own point selection algorithm.  
            input_points, input_labels = generate_input_points(  
                num_positive=self.num_positive,  
                num_negative=self.num_negative,  
                mask=ground_truth_mask,  
                dynamic_distance=True,  
                erode=self.erode,  
            )  
            input_points = input_points.astype(float).tolist()  
            input_labels = input_labels.tolist()  
            input_labels = [[x] for x in input_labels]

            # Prepare the image and prompt for the model.  
            inputs = self.processor(  
                input_image,  
                input_points=input_points,  
                input_labels=input_labels,  
                return_tensors="pt"  
            )

            # Remove batch dimension which the processor adds by default.  
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}  
            inputs["input_labels"] = inputs["input_labels"].squeeze(1)

            return inputs

    def get_input_bbox(self,mask, perturbation=0):
        # Find minimum mask bounding all included mask points.
        y_indices, x_indices = np.where(mask > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        if perturbation:  # Add perturbation to bounding box coordinates.
            H, W = mask.shape
            x_min = max(0, x_min + np.random.randint(-perturbation, perturbation))
            x_max = min(W, x_max + np.random.randint(-perturbation, perturbation))
            y_min = max(0, y_min + np.random.randint(-perturbation, perturbation))
            y_max = min(H, y_max + np.random.randint(-perturbation, perturbation))

        bbox = [x_min, y_min, x_max, y_max]
        return bbox

    def _getitem_bbox(self, input_image, ground_truth_mask):  
        # Get bounding box prompt.  
        bbox = self.get_input_bbox(ground_truth_mask, perturbation=self.perturbation)

        # Prepare the image and prompt for the model.  
        inputs = self.processor(input_image, input_boxes=[[bbox]], return_tensors="pt")  
        inputs = {k: v.squeeze(0) for k, v in inputs.items()} # Remove batch dimension which the processor adds by default.

        return inputs
    
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


from transformers import SamProcessor  
from torch.utils.data import DataLoader

def get_model_name(model_size):
    if not (model_size == "base" or model_size == "huge"):
        raise ValueError(f'Unknown model size "{model_size}".')

    return f"facebook/sam-vit-{model_size}"

def get_datasets(
    model_size,
    **kwargs,
):
    model_name = get_model_name(model_size)
    processor = SamProcessor.from_pretrained(model_name)

    train_dataset = SAMDataset(
        dataset=training_data, 
        processor=processor, 
        **kwargs,
    )
    validation_dataset = SAMDataset(
        dataset=validation_data,
        processor=processor, 
        **kwargs,
    )

    return train_dataset, validation_dataset

def get_dataloaders(
    model_size = "base", 
    batch_size = 7, 
    prompt_type = PromptType.CONTROL_POINTS,
    num_positive = 3,
    num_negative = 0,
    erode = True,
    multi_mask = "mean",
    perturbation = 10,
    image_size = (256, 256),
    mask_size = (256, 256),
    shuffle_train = True,
):
    model_name = get_model_name(model_size)
    processor = SamProcessor.from_pretrained(model_name)

    train_dataset, validation_dataset = get_datasets(
        model_size=model_size,
        prompt_type=prompt_type,
        num_positive=num_positive,
        num_negative=num_negative,
        erode=erode,
        perturbation=perturbation,
        image_size=image_size,
        mask_size=mask_size,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False) # Take note that batch size is always 1 here!

    return train_dataloader, validation_dataloader


import monai
import torchvision.ops as ops
import torch.nn.functional as F

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


def calculate_loss(
    outputs,
    ground_truth_masks: torch.Tensor, 
    prompt_type: PromptType, 
    loss_fn,
    multi_mask = None,
):
    predicted_masks = outputs.pred_masks.squeeze(1)

    if prompt_type == PromptType.CONTROL_POINTS:
        if multi_mask == "mean":
            predicted_masks = predicted_masks.mean(dim=1)
        
        if multi_mask == "best":
            best_index = torch.argmax(outputs.iou_scores, dim=1).squeeze(1)
            batch_indices = torch.arange(predicted_masks.shape[0])
            predicted_masks = predicted_masks[batch_indices, best_index]

    upsampled_pred = upsample_mask(predicted_masks.squeeze(1), ground_truth_masks.shape[1:]) # Slice off the batch dimension.
    return loss_fn(upsampled_pred, ground_truth_masks)

def forward_pass(model: SamModel, batch, prompt_type: PromptType):
    if prompt_type == PromptType.CONTROL_POINTS:
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_points=batch["input_points"],
            input_labels=batch["input_labels"],
            multimask_output=False,
        )
    elif prompt_type == PromptType.BOUNDING_BOX:
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_boxes=batch["input_boxes"],
            multimask_output=False,
        )
    return outputs

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

def save_model_checkpoint(
        accelerator,
        checkpoint_path,
        model: SamModel,
        optimizer: Optimizer,
        scheduler,
        epoch: int, # TODO: This is redundant, could be calculated from train_history.
        train_history,
        best_loss: float,
        train_losses,
        validation_losses,
        loss_config,
        model_descriptor: str = None,
    ):
    # accelerator.print(f"Saving checkpoint for epoch {epoch}...")
    model = accelerator.unwrap_model(model)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "train_history": train_history.get_history_list(),
        "loss_config": loss_config.names,
    }
    if (model_descriptor): checkpoint["descriptor"] = model_descriptor
    torch.save(checkpoint, checkpoint_path)


dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
def dice_loss_fn(pred_mask, ground_truth):
    return dice_loss(pred_mask, ground_truth)

class LossFunctionsConfig():
    def __init__(self, funcs, track_iou=False):
        self.names = [item[0] for item in funcs]
        self.funcs = [item[1] for item in funcs]
        self.track_iou = track_iou
default_loss_config = LossFunctionsConfig([("dice", dice_loss_fn)])


import gc

def gather_scores(accelerator, scores):
    all_scores = accelerator.gather(scores)
    mean_scores = torch.mean(all_scores, dim=0) if accelerator.num_processes > 1 else all_scores
    return mean_scores

def evaluate_model(model: SamModel, dataloader: DataLoader, device, loss_config: LossFunctionsConfig, loop) -> torch.Tensor:
    model.eval()
    val_losses: list[torch.Tensor] = []

    loop.n = 0 # Loop Reset
    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(): # Forward Pass
            outputs = forward_pass(model, batch, dataloader.dataset.prompt_type)

        # Compute Loss
        ground_truth_masks = batch["ground_truth_mask"].float()
        trained_loss = calculate_loss(
            outputs, 
            ground_truth_masks, 
            dataloader.dataset.prompt_type, 
            loss_config.funcs[0], 
            dataloader.dataset.multi_mask,
        )

        batch_losses = []
        for idx, loss_fn in enumerate(loss_config.funcs):
            loss = calculate_loss(outputs, ground_truth_masks, dataloader.dataset.prompt_type, loss_fn, dataloader.dataset.multi_mask)
            batch_losses.append(loss.item())
        val_losses.append(batch_losses)

        # Update Progress Bar
        loop.set_description(f"Eval Loss: {trained_loss.item():.4f}")
        loop.update(1)
    # END FOR

    # Avoid Memory Overload
    # TODO: Do we need this? Is this the right place for this?
    gc.collect()
    torch.cuda.empty_cache()

    return torch.mean(torch.Tensor(val_losses), dim=0)


####################
# main training loop
####################

# huggingface accelerate might work on multiple AWS instances?
# otherwise replace
from transformers import SamModel
from torch.optim import Optimizer
from accelerate import Accelerator
from accelerate.utils import set_seed, tqdm
from torch.optim.lr_scheduler import LinearLR
from torch.optim import AdamW

def training_loop(
        model_size="base",
        optimizer_name="AdamW",
        loss_config:LossFunctionsConfig=default_loss_config,
        learning_rate:float=1e-5,
        weight_decay:float=1e-5,
        batch_size:int=8,
        prompt_args:dict={},
        num_epochs=300,
        mixed_precision="fp16",
        seed:int=42,
        load_checkpoint:str=None,
        model_path_name:str=None,
        dataset_name = 'v1' # arbitrary tag for now
    ):
    global model_descriptor, best_checkpoint_path, train_losses, validation_losses, iou_scores

    set_seed(seed)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    train_dataloader, validation_dataloader = get_dataloaders(model_size, batch_size, **prompt_args)
    accelerator.print(f"Getting dataloaders for {model_size} model with prompt configuration: {prompt_args}")

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

    # Get model descriptor and save path.
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
    path_name = model_path_name or model_descriptor 
    best_checkpoint_path = "./checkpoints/" + path_name + ".pth"
    last_checkpoint_path = "./checkpoints/last_" + path_name + ".pth"
    accelerator.print("Using save checkpoint path:", best_checkpoint_path)

    if load_checkpoint is not None:
        model, optimizer, lr_scheduler, other = load_model_checkpoint(accelerator, load_checkpoint, model, optimizer, lr_scheduler)
        best_loss = other["best_loss"]
        train_losses = other["train_losses"]
        validation_losses = other["validation_losses"]
        train_history = other["train_history"]
        epoch = other["epoch"]

    train_history.start_session(
        dataset_name,
        optimizer_name,
        loss_config.names[0],
        learning_rate,
        weight_decay,
        batch_size,
        prompt_args,
    )

    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )

    # =============================== #
    # Main training loop starts here: #

    epoch_loop = tqdm(total=num_epochs, position=epoch, leave=False)
    batch_loop = tqdm(total=len(train_dataloader), position=0, leave=True)
    validation_loop = tqdm(total=len(validation_dataloader), position=0, leave=True)
    model.train()

    while epoch < num_epochs:
        epoch_losses = []
        prompt_type = train_dataloader.dataset.prompt_type

        batch_loop.n = 0 # Loop Reset
        for idx, batch in enumerate(train_dataloader):
            # Forward Pass
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            outputs = forward_pass(model, batch, prompt_type)

            # Compute Loss
            ground_truth_masks = batch["ground_truth_mask"].float()
            train_loss = calculate_loss(outputs, ground_truth_masks, prompt_type, loss_config.funcs[0], multi_mask=prompt_args["multi_mask"])
            
            batch_losses = []
            for idx, loss_fn in enumerate(loss_config.funcs):
                loss = calculate_loss(outputs, ground_truth_masks, prompt_type, loss_fn, multi_mask=prompt_args["multi_mask"])
                batch_losses.append(loss.item())
            epoch_losses.append(batch_losses)

            # Backward Pass & Optimizer Step
            optimizer.zero_grad()
            accelerator.backward(train_loss)
            optimizer.step()
            lr_scheduler.step()

            batch_loop.set_description(f"Train Loss: {train_loss.item():.4f}")
            batch_loop.update(1)
        # END FOR

        accelerator.wait_for_everyone()
        # TODO: Verify this entire section works in a distributed training environment.
        
        local_val_losses = evaluate_model(
            model, 
            validation_dataloader, 
            accelerator.device, 
            loss_config,
            validation_loop,
        )
        epoch_val_losses = gather_scores(accelerator, local_val_losses)
        validation_losses.append(epoch_val_losses)

        local_mean_losses = torch.mean(torch.Tensor(epoch_losses), dim=0)
        mean_losses = gather_scores(accelerator, local_mean_losses)
        train_losses.append(mean_losses)

        validation_loss = epoch_val_losses[0] if epoch_val_losses.shape else epoch_val_losses.item()
        train_history.update_session(epoch)
        if accelerator.is_main_process:
            if validation_loss < best_loss:
                save_model_checkpoint(
                    accelerator,
                    best_checkpoint_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    train_history,
                    validation_loss,
                    train_losses,
                    validation_losses,
                    loss_config,
                    model_descriptor=model_descriptor,
                )
                best_loss = validation_loss
        # END IF

        epoch_loop.set_description(f"Best Loss: {best_loss:.4f}")
        epoch_loop.update(1)
        epoch += 1
    # END WHILE

    # Save the last checkpoint, useful for experiments to see the whole loss history.
    if accelerator.is_main_process:
        save_model_checkpoint(
            accelerator,
            last_checkpoint_path,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            train_history,
            validation_loss,
            train_losses,
            validation_losses,
            loss_config,
            model_descriptor=model_descriptor,
        )


import time
from datetime import datetime, timezone, timedelta

def get_current_time():
    pst = timezone(timedelta(hours=-8), "PST")
    current_datetime = datetime.now(pst)
    return current_datetime.strftime("%H:%M %m/%d/%Y")

print(f"Training started at {get_current_time()}")

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
    "perturbation": 10,
}

args = (
    "base",  # model_size
    "AdamW",  # optimizer_name
    loss_config,  # loss_config
    7e-6,  # learning_rate
    2e-4,  # weight_decay
    5,  # batch_size (batch size per process is batch_size / num_processes)
    prompt_args,  # prompt_args
    50,  # num_epochs
    "fp16",  # mixed_precision ("no" for full precision)
    42,  # seed
    None,  # load_checkpoint (string path to checkpoint or None)
    None,  # model_path_name (override model path or None)
)
training_loop(args, num_processes=1)
print(f"Training ended at {get_current_time()}")