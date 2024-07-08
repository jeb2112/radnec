
import torch
import argparse
import json
import os
import re
import copy
import numpy as np
import nibabel as nb
from typing import Any, Dict, List
import PIL
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

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
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image,origin='lower')
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

# load a single nifti file
def loadnifti(self,t1_file,dir=None,type=None):
    img_arr_t1 = None
    if dir is None:
        dir = self.studydir
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

# compute the bounding hull from a mask. 
def _compute_hull_from_mask(mask, original_size=None, hull_extension=0):
    mask2 = np.copy(mask)
    if hull_extension > 0:
        for h in range(hull_extension):
            mask2 = binary_dilation(mask2)
    coords = np.transpose(np.array(np.where(mask2 == 1)))
    hull = ConvexHull(coords)
    hcoords = np.transpose(np.vstack((coords[hull.vertices,1],coords[hull.vertices,0])))
    if False:
        plt.figure(7)
        plt.cla()
        plt.imshow(mask)
        plt.plot(hcoords[:,0],hcoords[:,1],'r.',markersize=5)
        plt.show(block=False)
        a=1
    return hcoords

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



def main(args: argparse.Namespace) -> None:
    
    
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    output_mode = "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    predictor = SamPredictor(sam)


    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        studydirs = [f for f in os.listdir(args.input) if os.path.isdir(os.path.join(args.input,f))]

    for s in studydirs:
        spath = os.path.join(args.input,s,'sam')
        if not os.path.exists(spath):
            continue
        files = os.listdir(spath)
        if len(files) == 0:
            continue
        # for ch,img in zip(['flair','t1+'],['WT','ET']):
        for ch,img in zip(['t1+'],['ET']):
            images = [ f for f in files if re.match('slice_[0-9]+_'+ch,f) ]
            images = sorted([os.path.join(spath, f) for f in images])
            masks = [ f for f in files if re.match('mask_[0-9]+_'+ch,f)]
            masks = sorted([os.path.join(spath, f) for f in masks])
            image_pil = PIL.Image.open(images[0])
            slicedim = int(image_pil.info['slicedim'])
            affine_enc = image_pil.info['affine'].encode()
            affine_dec = affine_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
            affine = np.reshape(np.frombuffer(affine_dec, dtype=np.float64),(4,4))
            sam_mask_point = np.zeros((slicedim,)+image_pil.size[-1::-1])
            sam_mask_box = np.zeros((slicedim,)+image_pil.size[-1::-1])
            sam_mask_boxpoint = np.zeros((slicedim,)+image_pil.size[-1::-1])

            # for m,t in zip([masks[0]],[images[0]]):
            for m,t in zip(masks,images):
                print(f"Processing '{t}'...")
                slice = int(re.search('slice_([0-9]+)',t).group(1))
                image_pil = PIL.Image.open(t)
                image = np.atleast_3d(np.array(image_pil)[:,:,:3])
                if image is None:
                    print(f"Could not load '{t}' as an image, skipping...")
                    continue
                mask = np.squeeze(plt.imread(m)[:,:,0]*1).astype(np.uint8)
                if mask is None:
                    print('could not load {} as a mask, skipping...'.format(m))
                    continue
                predictor.set_image(image)

                bbox = _compute_box_from_mask(mask,box_extension=3)
                point = np.array(list(map(int,np.mean(np.where(mask),axis=1)))) 
                ros = np.array(np.where(image[:,:,0]))
                distances = np.sqrt(np.power(ros[0]-point[0],2)+np.power(ros[1]-point[1],2))
                arg = np.argsort(np.abs(distances - np.median(distances)))[0]
                point_bg = np.atleast_2d(np.flip(ros[:,arg])) # flip for xy
                point = np.atleast_2d(np.flip(point))
                convexhull = _compute_hull_from_mask(mask,hull_extension=2)
                convexhull_points = np.concatenate((np.atleast_2d(point),convexhull),axis=0)
                convexhull_labels = np.array([1]+[0]*len(convexhull))
                
                # points prompt
                # points = np.concatenate((point,point_bg),axis=0)
                # points_label = np.array([1,0]) # foreground,background
                masks, scores, _ = predictor.predict(
                    point_coords=convexhull_points,
                    point_labels=convexhull_labels,
                    multimask_output=True,
                )
                mask_sizes = np.zeros_like(scores)
                for i,m in enumerate(masks):
                    mask_sizes[i] = len(np.where(m)[0])
                min_mask = np.where(mask_sizes == min(mask_sizes))[0]
                if np.argmax(scores) == min_mask:
                    sam_mask_point[slice] = np.copy(masks[np.argmax(scores),:,:])
                else:
                    sam_mask_point[slice] = np.copy(masks[min_mask,:,:])

                # box prompt
                sam_mask_box[slice], _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box = bbox[None,:],
                    multimask_output=False,
                )

                # box points prompt
                sam_mask_boxpoint[slice],_,_ = predictor.predict(
                    point_coords=convexhull_points,
                    point_labels=convexhull_labels,
                    box = bbox[None,:],
                    multimask_output=False,
                )
                if True:
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
                    ax[2].set_title('box')
                    ax[3].cla()
                    ax[3].imshow(image,origin='lower')
                    show_mask(sam_mask_boxpoint[slice],ax[3])
                    ax[3].set_title('box-points')
                    plt.show(block=False)
                    a=1

            writenifti(sam_mask_point,os.path.join(args.output,s,'{}_sam_points_processed.nii'.format(img)),type=np.uint8,affine=affine)
            writenifti(sam_mask_box,os.path.join(args.output,s,'{}_sam_box_processed.nii'.format(img)),type=np.uint8,affine=affine)
            writenifti(sam_mask_boxpoint,os.path.join(args.output,s,'{}_sam_boxpoints_processed.nii'.format(img)),type=np.uint8,affine=affine)

            opath = spath


    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
