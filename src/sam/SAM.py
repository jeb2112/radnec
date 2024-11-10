# script for loading checkpoint results
# and running a prediction
#  
# from sam fine-tuning with BraTS2024

import numpy as np
import os
import re
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import f1_score, precision_score
from scipy.spatial.distance import dice,directed_hausdorff
import copy
from cProfile import Profile
from pstats import SortKey,Stats

import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import nibabel as nb
import PIL 

from transformers import SamModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.sam.SAMProcessing import SAMProcessing
from src.sam.SAMMisc import *


class SAM():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######################
    # prediction functions
    ######################

    def upsample_mask(self,pred_mask, target_size):
        return F.interpolate(pred_mask.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)

    def forward_pass(self,model: SamModel, batch, prompt_type: PromptType):
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

    def calc_datapoint_metrics(self,y_true_mask, y_pred_mask):
        y_true = y_true_mask.ravel()
        y_pred = y_pred_mask.ravel()
        hd =  max(directed_hausdorff(np.array(np.where(y_true_mask)).T,np.array(np.where(y_pred_mask)).T)[0],
                                                            directed_hausdorff(np.array(np.where(y_pred_mask)).T,np.array(np.where(y_true_mask)).T)[0])
        if hd == np.Inf:
            hd = np.NaN
        return {
            # "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "dice": 1-dice(y_true,y_pred),
            "hausdorff": hd

        }

    def predict_model(self,model: SamModel, batch, prompt_args, confidence_threshold=0.5, device="cuda"):

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = self.forward_pass(model, batch, prompt_args["prompt_type"])

        # Squeeze the 0th dim since validation dataloader always has batch size 1, and
        # attempt to squeeze the 1st dim since bbox predictions only have 1 output mask.
        raw_predicted_masks = outputs.pred_masks.squeeze((0, 1)).cpu()
        ground_truth_mask = batch["ground_truth_mask"].squeeze(0).float().cpu()

        predicted_mask = torch.sigmoid(self.upsample_mask(raw_predicted_masks, ground_truth_mask.shape)).squeeze()
        raw_mask = predicted_mask.cpu().numpy()
        mask = (raw_mask > confidence_threshold).astype(np.uint8) * 255
        # metrics = calc_datapoint_metrics(ground_truth_mask.numpy().astype(np.uint8), mask)
        metrics = None
        
        mask_comb = None
        if False:
            bbox = batch['input_boxes'][0][0].cpu().numpy().astype('int')
            bbox_shape = tuple(batch['reshaped_input_sizes'][0].cpu().numpy())
            mask_comb = predict_plot(ground_truth_mask.numpy().astype(np.uint8),mask,bbox,bbox_shape)

        return raw_mask, mask, metrics, mask_comb


    def predict_metrics(self,model: SamModel, dataloader: DataLoader, prompt_args, datadir=None, device="cuda"):
        model.to(device)
        model.eval()

        metrics = []    
        
        for idx, batch in enumerate(dataloader):
            _, mask, batch_metrics, _ = self.predict_model(model, batch, prompt_args)
            metrics.append(batch_metrics)

            if datadir is not None:
                ofile = os.path.join(datadir,'predictions','pred_mask_' + str(idx).zfill(5) + '.png')
                imsave(ofile,mask,check_contrast=False)

        return metrics

    def predict_plot(self,gt,pred,bbox,shape):
        comb_mask = np.zeros(shape+(3,),dtype='uint8')
        gt = resize(gt,shape,order=0).astype('uint8') * 1
        pred = resize(pred,shape,order=0).astype('uint8') * 1
        comb_mask[gt>0,0] = 255 # red
        comb_mask[pred>0,2] = 255 # blue
        rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=comb_mask.shape)
        comb_mask[rr,cc,1] = 255
        return comb_mask

    ################
    # misc functions
    ################
            
    def set_slice(self,idx,img_arr_3d,img_arr_2d,orient):
        if orient == 'ax':
            img_arr_3d[idx] = copy.deepcopy(img_arr_2d)
        elif orient == 'sag':
            img_arr_3d[:,:,idx] = copy.deepcopy(img_arr_2d)
        elif orient == 'cor':
            img_arr_3d[:,idx,:] = copy.deepcopy(img_arr_2d)

    def load_model_checkpoint(self,checkpoint_path, model: SamModel):
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Model checkpoint loaded from: {checkpoint_path}")

        return model

    # load a single nifti file
    def loadnifti(self,t1_file,dir,type=None):
        img_arr_t1 = None
        try:
            img_nb_t1 = nb.load(os.path.join(dir,t1_file))
        except IOError as e:
            print('Can\'t import {}'.format(t1_file))
            return None,None
        nb_header = img_nb_t1.header.copy()
        # nibabel convention will be transposed to sitk convention
        img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
        if type is not None:
            img_arr_t1 = img_arr_t1.astype(type)
        affine = img_nb_t1.affine
        return img_arr_t1,affine

    # write a single nifti file. use uint8 for masks 
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force "{}"'.format(filename))


    #############
    # script main
    #############

    def main(self,checkpoint=None,input=None,tag=None,layer=None,orient=None,prompt='bbox',
             pretrained=None,output=None,debug=False):

        prompt_args = {'point':{    # Control points version:
            "prompt_type": PromptType.CONTROL_POINTS,
            "num_positive": 1, # hard-coded for single point now
            "num_negative": 0, # no background points
            "erode": True,
            "multi_mask": "mean",
            },
            'bbox':{ # Bounding boxes version:
            "prompt_type": PromptType.BOUNDING_BOX,
            "perturbation": 0,
            "padding": 3,
            "multi_mask": None
            }
        }

        if pretrained is not None:
            # getting a path error on this syntax
            model = SamModel.from_pretrained(pretrained)
        else:
            # this loads from cache after 1st download
            model = SamModel.from_pretrained(f"facebook/sam-vit-base")
        if checkpoint is not None:
            checkpoint_data = torch.load(checkpoint)
            model.load_state_dict(checkpoint_data["model_state_dict"])
        model.to(self.device)


        sfiles = os.listdir(input)
        studydirs = [s for s in sfiles if os.path.isdir(os.path.join(input,s,'sam',orient,'images'))]
        if len(studydirs) == 0:
            raise FileNotFoundError('No images sub-directory found')
        for s in studydirs: # only coded for 1 actual dir right now

            # prepare output dir. When orthogonal planes are being predicted, 'ax' will be first.
            if output is None:
                outputdir = os.path.join(input,s,'sam','predictions_nifti')
            else:
                outputdir = output
            # if args.orient == 'ax':
            #     if os.path.exists(outputdir):
            #         shutil.rmtree(outputdir)
            #     os.mkdir(outputdir)
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)

            spath = os.path.join(input,s,'sam',orient)
            if not os.path.exists(spath):
                raise NotADirectoryError
            files = os.listdir(spath)
            if len(files) == 0:
                raise FileNotFoundError
            
            eval_datadir = {'test':spath}
            samp = SAMProcessing(eval_datadir,model_size='base',prompt_type=prompt_args[prompt]['prompt_type'])

            if debug == False:
                res = self.predict_metrics(model,samp.dataloaders['test'],prompt_args[prompt],datadir=spath)

            # gather the sam-predicted 2d slices into a nifti volume
            # with the torch DataLoader, the output masks for a set of image files are just in 
            # plain index order. map these back to the slice positions according to the filenames
            # of the input image files. it is thus assumed that the DataLoader is processing the image
            # files in a sorted order since it is batchsize=1 and no shuffling (to be verified)
            img_files = os.listdir(os.path.join(spath,'images'))
            images = sorted([ f for f in img_files if re.match('img.*slice_[0-9]{3}',f) ])
            image_pil = PIL.Image.open(os.path.join(spath,'images',images[0]))
            # for lack of any better way to pass to this standalone script, these values are encoded in the .png header
            # when exported form the viewer. this script should be a method in the CreateSAMROIFrame.class
            slicedim = int(image_pil.info['slicedim'])
            affine_enc = image_pil.info['affine'].encode()
            affine_dec = affine_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
            affine = np.reshape(np.frombuffer(affine_dec, dtype=np.float64),(4,4))
            if orient == 'ax':
                sam_predict = np.zeros((slicedim,)+image_pil.size[-1::-1],dtype='uint8')
            elif orient == 'sag':
                sam_predict = np.zeros(image_pil.size[-1::-1]+(slicedim,),dtype='uint8')
            elif orient == 'cor': 
                sam_predict = np.zeros((image_pil.size[-1],slicedim,image_pil.size[0]),dtype='uint8')

            pred_files = os.listdir(os.path.join(spath,'predictions'))
            pred_masks = sorted([ f for f in pred_files if re.match('pred_mask',f) ])
            for i,m in zip(images,pred_masks):
                slice = int(re.search('slice_([0-9]{3})',i).group(1))
                mask_pil = PIL.Image.open(os.path.join(spath,'predictions',m))
                mask = skimage.transform.resize(np.array(mask_pil),image_pil.size[-1::-1],order=0)
                self.set_slice(slice,sam_predict,mask,orient)
                # sam_predict[slice] = mask

            if np.max(sam_predict) > 1:
                sam_predict[np.where(sam_predict)] = 1
            fname = '{}_sam_{}_{}_{}.nii'.format(layer,prompt,tag,orient)
            self.writenifti(sam_predict,os.path.join(outputdir,fname),type=np.uint8,affine=affine)

            # clean up working directories
            if True:
                shutil.rmtree(spath)
                for d in ['images','prompts','predictions']:
                    os.makedirs(os.path.join(spath,d),exist_ok=True)


