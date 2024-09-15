# script for loading sam checkpoint results and running predictions
# uses the huggingface classes

import numpy as np
import argparse
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score
from scipy.spatial.distance import dice,directed_hausdorff
from collections import defaultdict
import json
import copy

import skimage
from skimage.io import imread,imsave
from skimage.transform import resize
import pandas as pd
import nibabel as nb
import PIL

from transformers import SamModel
import torch
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# prediction functions
######################

def _process_box(box, shape, original_size=None, box_extension=0):
    if box_extension == 0:  # no extension
        extension_y, extension_x = 0, 0
    elif box_extension >= 1:  # extension by a fixed factor
        extension_y, extension_x = box_extension, box_extension
    else:  # extension by fraction of the box len
        len_y, len_x = box[2] - box[0], box[3] - box[1]
        extension_y, extension_x = box_extension * len_y, box_extension * len_x

    box = np.array([
        max(box[1] - extension_x, 0), max(box[0] - extension_y, 0),
        min(box[3] + extension_x, shape[1]), min(box[2] + extension_y, shape[0]),
    ])

    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box = trafo.apply_boxes(box[None], (256, 256)).squeeze()
    return box

# compute the bounding box from a mask. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box_from_mask(mask, original_size=None, box_extension=0):
    coords = np.where(mask == 1)
    min_y, min_x = coords[0].min(), coords[1].min()
    max_y, max_x = coords[0].max(), coords[1].max()
    box = np.array([min_y, min_x, max_y + 1, max_x + 1])
    return _process_box(box, mask.shape, original_size=original_size, box_extension=box_extension)

def upsample_mask(pred_mask, target_size):
    return F.interpolate(pred_mask.unsqueeze(1), size=target_size, mode="nearest").squeeze(1)

def forward_pass(model: SamModel, batch, prompt_type: str):
    if prompt_type == 'point':
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_points=batch["input_points"],
            input_labels=batch["input_labels"],
            multimask_output=False,
        )
    elif prompt_type == 'bbox':
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_boxes=batch["input_boxes"],
            multimask_output=False,
        )
    return outputs

def calc_datapoint_metrics(y_true_mask, y_pred_mask):
    y_true = y_true_mask.ravel()
    y_pred = y_pred_mask.ravel()
    hd =  max(directed_hausdorff(np.array(np.where(y_true_mask)).T,np.array(np.where(y_pred_mask)).T)[0],
                                                        directed_hausdorff(np.array(np.where(y_pred_mask)).T,np.array(np.where(y_true_mask)).T)[0])
    if hd == np.Inf:
        hd = np.NaN
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "dice": 1-dice(y_true,y_pred),
        "hausdorff": hd

    }

# m = filename of a mask of the target in the image to be segmented, can be used to make a bounding box prompt
# t = filename of target image to be segmented
# predictor = instance of facebook SamPredictor class
def predict_model(model: SamModel, batch, confidence_threshold=0.5):

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = forward_pass(model, batch, prompt_args["prompt_type"])

    # Squeeze the 0th dim since validation dataloader always has batch size 1, and
    # attempt to squeeze the 1st dim since bbox predictions only have 1 output mask.
    raw_predicted_masks = outputs.pred_masks.squeeze((0, 1)).cpu()
    ground_truth_mask = batch["ground_truth_mask"].squeeze(0).float().cpu()

    predicted_mask = torch.sigmoid(upsample_mask(raw_predicted_masks, ground_truth_mask.shape)).squeeze()
    raw_mask = predicted_mask.cpu().numpy()
    mask = (raw_mask > confidence_threshold).astype(np.uint8)
    metrics = calc_datapoint_metrics(ground_truth_mask.numpy().astype(np.uint8), mask)
    
    mask_comb = None
    if True:
        bbox = batch['input_boxes'][0][0].cpu().numpy().astype('int')
        bbox_shape = tuple(batch['reshaped_input_sizes'][0].cpu().numpy())
        mask_comb = predict_plot(ground_truth_mask.numpy().astype(np.uint8),mask,bbox,bbox_shape)

    return raw_mask, mask, metrics, mask_comb

# m = filename of a mask of the target in the image to be segmented, can be used to make a bounding box prompt
# t = filename of target image to be segmented
# predictor = instance of facebook SamPredictor class
def predict_metrics(m,t,model: SamModel):
    model.to(device)
    model.eval()

    iterCount = 0
    metrics = []    
    
    batch = {}
    for idx, batch in enumerate(dataloader):
        _, _, batch_metrics, mask_comb = predict_model(model, batch)
        metrics.append(batch_metrics)

        if mask_comb is not None and datadir is not None:
            ofile = os.path.join(datadir,'datacheck','mask_comb_' + str(idx).zfill(4) + '.png')
            imsave(ofile,mask_comb,check_contrast=False)

    joined = defaultdict(list)
    for metric in metrics:
        for key in metric:
            joined[key].append(metric[key])

    return joined

def predict_plot(gt,pred,bbox,shape):
    comb_mask = np.zeros(shape+(3,),dtype='uint8')
    gt = resize(gt,shape,order=0).astype('uint8') * 1
    pred = resize(pred,shape,order=0).astype('uint8') * 1
    comb_mask[gt>0,0] = 255 # red
    comb_mask[pred>0,2] = 255 # blue
    rr,cc = skimage.draw.rectangle_perimeter(bbox[1::-1],end=bbox[:1:-1],shape=comb_mask.shape)
    comb_mask[rr,cc,1] = 255
    return comb_mask

# main function to segment the prompt, with filename args
# m = filename of a mask of the target in the image to be segmented, can be used to make a bounding box prompt
# t = filename of target image to be segmented
# model = instance of huggingface SamModel class
def run_sam(m,t,prompt,model):
    print(f"Processing '{t}'...")
    slice = int(re.search('slice_([0-9]+)',t).group(1))
    image_pil = PIL.Image.open(t)
    image = np.atleast_3d(np.array(image_pil)[:,:,:3])
    if image is None:
        print(f"Could not load '{t}' as an image, skipping...")
        return
    mask = np.squeeze(plt.imread(m)[:,:,0]*1).astype(np.uint8)
    if mask is None:
        print('could not load {} as a mask, skipping...'.format(m))
        return

    if prompt == 'bbox':
        bbox = _compute_box_from_mask(mask,box_extension=3)
    point = np.array(list(map(int,np.mean(np.where(mask),axis=1)))) 
    ros = np.array(np.where(image[:,:,0]))
    distances = np.sqrt(np.power(ros[0]-point[0],2)+np.power(ros[1]-point[1],2))
    arg = np.argsort(np.abs(distances - np.median(distances)))[0]
    point_bg = np.atleast_2d(np.flip(ros[:,arg])) # flip for xy
    point = np.atleast_2d(np.flip(point))
    if False:
        # optional. add some background points using a convex hull around the mask
        convexhull = _compute_hull_from_mask(mask,hull_extension=2)
        convexhull_points = np.concatenate((np.atleast_2d(point),convexhull),axis=0)
        convexhull_labels = np.array([1]+[0]*len(convexhull))

    # points prompt
    # points = np.concatenate((point,point_bg),axis=0)
    points = point # no background, just single foreground point
    # points_label = np.array([1,0]) # foreground,background
    points_labels = np.array([1]) # foreground only
    batch = {}
    _, _, batch_metrics, mask_comb = predict_model(model, batch)


    mask_sizes = np.zeros_like(scores)
    for i,m in enumerate(masks):
        mask_sizes[i] = len(np.where(m)[0])
    min_mask = np.where(mask_sizes == min(mask_sizes))[0]
    if len(min_mask > 0):
        min_mask = min_mask[0]
    if np.argmax(scores) == min_mask:
        sam_mask_point = np.copy(masks[np.argmax(scores),:,:])
    else:
        sam_mask_point = np.copy(masks[min_mask,:,:])

    if False:
        fig, ax = plt.subplots(1, 4, num=7, figsize=(10, 2.5),sharex=True,sharey=True)
        ax[0].cla()
        ax[0].imshow(image,origin='lower')
        show_box(bbox, ax[0])
        show_points(convexhull_points,convexhull_labels,ax[0],marker_size=10)
        ax[0].set_title("Input Image and Bounding Box")
        ax[1].cla()
        ax[1].imshow(image,origin='lower')
        show_mask(sam_mask_point[slice], ax[1])
        ax[1].set_title('points')
        ax[2].cla()
        ax[2].imshow(image,origin='lower')
        show_mask(sam_mask_box[slice],ax[2])
        ax[3].cla()
        ax[3].imshow(image,origin='lower')
        show_mask(sam_mask_boxpoint[slice],ax[3])
        ax[3].set_title('box-points')
        plt.show(block=False)
        a=1

    return (sam_predict)

################
# misc functions
################
    
# load a single nifti file
def loadnifti(t1_file,dir,type=None):
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
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
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

def main(args):

    prompt_args = {'point':{    # Control points version:
        "prompt_type": 'point',
        "num_positive": 1, # hard-coded for single point now
        "num_negative": 0, # no background points
        "erode": True,
        "multi_mask": "mean",
        },
        'bbox':{ # Bounding boxes version:
        "prompt_type": 'bbox',
        "perturbation": 0,
        "padding": 3,
        "multi_mask": None
        }
    }

    model = SamModel.from_pretrained(args.pretrained)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    datadir = "C:\\Users\\Chris Heyn Lab\\data\\brats2024\\raw"
    test_datadir = os.path.join(datadir,'test_slice')
    datadirs = {'test':test_datadir}

    studydir = os.listdir(args.input)
    spath = os.path.join(args.input,studydir,'sam')
    if not os.path.exists(spath):
        raise NotADirectoryError
    files = os.listdir(spath)
    if len(files) == 0:
        raise FileNotFoundError
    # for ch,img in zip(['flair','t1+'],['WT','ET']):
    # by definition SAM output from t1+ input is TC, not just ET
    for ch,img in zip(['t1+'],['TC']):
        images = [ f for f in files if re.match('slice_[0-9]+_'+ch,f) ]
        images = sorted([os.path.join(spath, f) for f in images])
        masks = [ f for f in files if re.match('mask_[0-9]+_'+ch,f)]
        masks = sorted([os.path.join(spath, f) for f in masks])
        image_pil = PIL.Image.open(images[0])
        slicedim = int(image_pil.info['slicedim'])
        affine_enc = image_pil.info['affine'].encode()
        affine_dec = affine_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
        affine = np.reshape(np.frombuffer(affine_dec, dtype=np.float64),(4,4))
        sam_predict = np.zeros((slicedim,)+image_pil.size[-1::-1])

        # for m,t in zip([masks[0]],[images[0]]):
        for m,t in zip(masks,images):
            print(f"Processing '{t}'...")
            res = run_sam(m,t,predictor,)
            res = predict_metrics(m,t,model)
            slice = int(re.search('slice_([0-9]+)',t).group(1))

            sam_predict[slice] = res

        writenifti(sam_predict,os.path.join(args.output,s,'{}_sam_{}_{}.nii'.format(img,args.tag,args.prompt)),type=np.uint8,affine=affine)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="The path to the SAM pretrained model.",
    ),
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="The path to the SAM fine-tuned checkpoint to use for mask generation.",
        default=None
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="bbox|point",
        default='bbox'
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--tag",type=str, default="",help="Tag word for file naming")
    args = parser.parse_args()
    main(args)
