# demo script to run Medsam inference

import torch
import argparse
import json
import os
import re
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import transform
import torch.nn.functional as F

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
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def prepare_image(image, resize_transform, device):
    if False:
        img_1024_tensor = (
            torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0).to(device.device)
        )
    if False: # scale to 1024 preserving aspect
        if image.dtype=='float32':
            image_temp = (image*255).astype(np.uint8)
            image_temp = resize_transform.apply_image(image_temp,mode='RGB')
            img_1024 = image_temp.astype(np.float32)
        else:
            img_1024 = resize_transform.apply_image(image)
    else: # scale to 1024 square. medsam image encoding seemed to throw error with aspect scaling
        img_1024 = transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True)

    imaget = torch.as_tensor(img_1024, device=device.device, dtype=torch.float32)
    return imaget.permute(2, 0, 1).unsqueeze(0).contiguous()
    
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

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


######
# main
######

def main(args: argparse.Namespace) -> None:
    
    amg_kwargs = get_amg_kwargs(args)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    predictor = SamPredictor(sam)

    print("Loading model...")
    # _ = sam.to(device=args.device)
    sam = sam.to(device=args.device)
    sam.eval()
    output_mode = "binary_mask"

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
        for ch in ['t1+','flair']:
            images = [ f for f in files if re.match('slice_[0-9]+_'+ch,f) ]
            images = sorted([os.path.join(spath, f) for f in images])
            masks = [ f for f in files if re.match('mask_[0-9]+_'+ch,f)]
            masks = sorted([os.path.join(spath, f) for f in masks])

            for m,t in zip([masks[7]],[images[7]]):
                print(f"Processing '{t}'...")
                image = np.atleast_3d(plt.imread(t)[:,:,:3].astype(np.float32))
                H, W, _ = image.shape
                if image is None:
                    print(f"Could not load '{t}' as an image, skipping...")
                    continue
                mask = (plt.imread(m)[:,:,:3]*1).astype(np.float32)
                if mask is None:
                    print('Could not load {} as a mask, skipping...'.format(m))
                    continue
                if True: # 
                    imaget = prepare_image(image,resize_transform,sam)
                else: # bbox from mask
                    image = torch.as_tensor(image, device=sam.device, dtype=torch.float32)
                    image = image.permute(2, 0, 1).contiguous()
                bbox = _compute_box_from_mask(mask,box_extension=1)
                if True: # square scaling
                    box_1024 = np.atleast_2d(bbox / np.array([W, H, W, H]) * 1024)
                else: # aspect scaling.
                    bboxt = torch.tensor(bbox,device=sam.device)
                    bboxt = resize_transform.apply_boxes_torch(bboxt,image.shape[:2])
                    box_1024 = bboxt.cpu().data.numpy()[0]

                with torch.no_grad():
                    image_embedding = sam.image_encoder(imaget)  # (1, 256, 64, 64)
                medsam_seg = medsam_inference(sam, image_embedding, box_1024, H, W)

                if False:
                    plt.figure(7)
                    plt.clf()
                    plt.imshow(imaget.data.cpu().data.numpy()[0])
                    bboxt_arr = bboxt.cpu().data.numpy()[0]
                    show_box(bboxt_arr,plt.gca())
                    show_mask(medsam_seg)

                base = os.path.basename(t)
                base = os.path.splitext(base)[0]
                save_base = os.path.join(args.output, base)
                if output_mode == "binary_mask":
                    os.makedirs(save_base, exist_ok=False)
                    write_masks_to_folder(masks, save_base)
                else:
                    save_file = save_base + ".json"
                    with open(save_file, "w") as f:
                        json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
