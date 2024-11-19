
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone, timedelta

from transformers import SamProcessor  
from torch.utils.data import DataLoader

from src.sam.SAMDataset import SAMDataset,PromptType

class SAMProcessing():
    def __init__(self,datadirs,model_dict=None,model_size='base',batch_size=1,**kwargs):
        self.datadirs = datadirs
        self.dkeys = self.datadirs.keys()
        self.model_size = model_size
        self.batch_size = batch_size
        self.model_dict = model_dict # full path to local model

        self.datasets = {}
        self.dataloaders = {}
        for d in self.dkeys:
            if d in ['test','validate','eval']:
                shuffle_train = False
            else:
                shuffle_train = True
            self.get_dataloader(d,shuffle_train=shuffle_train,**kwargs)

        return

    def get_model_name(self,model_size='base'):
        if not (model_size == "base" or model_size == "huge"):
            raise ValueError(f'Unknown model size "{model_size}".')

        return f"facebook/sam-vit-{model_size}"

    def get_datasets(self,
        dkey,
        **kwargs,
    ):
        model_name = self.get_model_name(self.model_size)
        try:
            processor = SamProcessor.from_pretrained(model_name,state_dict=self.model_dict,do_rescale=False)
        except OSError as e:
            processor = SamProcessor.from_pretrained(model_name,state_dict=self.model_dict,do_rescale=False)

        self.datasets[dkey] = SAMDataset(
            datadir=self.datadirs[dkey], 
            processor=processor, 
            **kwargs,
        )
 
    def get_dataloader(self,
        dkey,
        prompt_type = PromptType.BOUNDING_BOX,
        num_positive = 3,
        num_negative = 0,
        erode = True,
        multi_mask = None,
        perturbation = 0,
        padding = 3,
        image_size = (256, 256),
        mask_size = (256, 256),
        shuffle_train = True,
    ):
        model_name = self.get_model_name(self.model_size)

        self.get_datasets(
            dkey,
            prompt_type=prompt_type,
            num_positive=num_positive,
            num_negative=num_negative,
            erode=erode,
            perturbation=perturbation,
            padding=padding,
            image_size=image_size,
            mask_size=mask_size,
        )

        self.dataloaders[dkey] = DataLoader(self.datasets[dkey], batch_size=self.batch_size, shuffle=shuffle_train)

