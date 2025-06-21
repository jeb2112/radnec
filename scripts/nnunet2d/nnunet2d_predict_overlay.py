# quick script to make overlays from base image, seg and prediction

import os
import cv2
import numpy as np
import imageio
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
import re
import shutil

# === Configuration ===

base_img_dir = "images"         # Base input images (grayscale or RGB)
gt_mask_dir = "masks_gt"        # Ground truth segmentation masks (uint8 with values like 0,1,2)
pred_mask_dir = "masks_pred"    # Predicted segmentation masks (same format as above)
results_dir = os.path.join(os.getenv('nnUNet_results'),'Dataset139_RadNec','nnUNetTrainerUnionDiceLoss__nnUNetPlans__2d','fold_0')
pred_mask_dir = os.path.join(results_dir,'dumpTs')
base_img_dir = os.path.join(os.getenv('nnUNet_raw'),'Dataset139_RadNec','imagesTs')
if False:
    gt_mask_dir = os.path.join(os.getenv('nnUNet_preprocessed'),'Dataset139_RadNec','gt_segmentations')
else:
    gt_mask_dir = os.path.join(os.getenv('nnUNet_raw'),'Dataset139_RadNec','labelsTs')
output_dir = os.path.join(results_dir,'dump_overlay')
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass
os.makedirs(output_dir, exist_ok=True)

# === Color map for masks ===
def colorize_mask(mask, colormap=cm.tab10):
    """Convert a label mask to RGB using a colormap"""
    norm_mask = mask.astype(np.float32) / mask.max() if mask.max() > 0 else mask
    return (colormap(norm_mask)[:, :, :3] * 255).astype(np.uint8)

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')  # Remove the '#' if it exists
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def label_mask_to_rgb(mask):
    defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    """Map specific label values to RGB colors"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 1] = hex_to_rgb(defcolors[0])    # Green for label 1
    rgb[mask == 2] = hex_to_rgb(defcolors[1])    # Red for label 2
    return rgb

# === Overlay helper ===
def overlay_mask(image, mask_rgb, alpha=0.4):
    return cv2.addWeighted(image, 1.0, mask_rgb, alpha, 0)

def overlay_mask_masked(image, mask_rgb, mask_labels, alpha=0.4):
    """Overlay the color mask only on the labeled pixels"""
    # Ensure mask is boolean for areas to overlay
    mask_bool = mask_labels > 0
    mask_bool_3c = np.stack([mask_bool]*3, axis=-1)

    # Copy the image so we don't modify in-place
    overlaid = image.copy().astype(np.float32)
    overlaid[mask_bool_3c] = (
        (1 - alpha) * image[mask_bool_3c].astype(np.float32) + 
        alpha * mask_rgb[mask_bool_3c].astype(np.float32)
    )
    return overlaid.astype(np.uint8)

# === Processing loop ===
files = os.listdir(base_img_dir)
files = sorted([f for f in files if '_0001.png' in f and ('val' in f or 'test' in f)])
for filename in files:
    fileroot = re.sub('(\_[0-9]{4})(\.png)',r'\2',filename)
    base_path = Path(base_img_dir) / filename
    gt_path = Path(gt_mask_dir) / fileroot
    pred_path = Path(pred_mask_dir) / fileroot

    if not (gt_path.exists() and pred_path.exists()):
        print(f"Skipping {filename}, missing GT or prediction")
        continue

    # Load base image
    base_img = imageio.imread(base_path)
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    # Load masks
    gt_mask = imageio.imread(gt_path)
    pred_mask = imageio.imread(pred_path)

    # Colorize
    gt_rgb = label_mask_to_rgb(gt_mask)
    pred_rgb = label_mask_to_rgb(pred_mask)

    # Overlay
    gt_overlay = overlay_mask_masked(base_img, gt_rgb, gt_mask)
    pred_overlay = overlay_mask_masked(base_img, pred_rgb, pred_mask)

    # Combine horizontally
    stacked = np.hstack((base_img, gt_overlay, pred_overlay))
    cv2.putText(stacked, 'ground truth', (200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(stacked, 'prediction', (400,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Save result
    out_path = Path(output_dir) / filename
    imageio.imwrite(out_path, stacked)
    print(f"Saved: {filename}")
