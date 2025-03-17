# some functions for creating overlay plots

from typing import Tuple, Union
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.draw import line_aa

# colors for ET,TC and WT layers
color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
)

# hard-coded convention for layers. nnunet is 4,2,1, BraTS is 3,2,1.
# this is a convenience definition for reverse lookup from integer back to dict keyword, for
# 'all' and 'both' in contour overlay, but it's awkward

# layerdict = {'ET':[3],'TC':[2],'WT':[1],'T2 hyper':[1],'all':[1,2,3],'both':[1,3]}
# layersdict = {1:'WT',2:'TC',3:'ET'}
layerdict = {'ET':[4],'TC':[2],'WT':[1],'T2 hyper':[1],'all':[1,2,4],'both':[1,4]}
layersdict = {1:'WT',2:'TC',4:'ET'}

def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))

# create blast overlay for raw or final smoothed ROI
def generate_blast_overlay(input_image: np.ndarray, segmentation: np.ndarray = None, contour: dict = None, layer: str = None,
                     mapping: dict = None,
                     color_cycle: Tuple[str, ...] = color_cycle,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d or 3d grayscale, with a multi-channel dimension prepended.
    Segmentation must be label map of same shape as 2d or 3d image (w/o color channels)
    Contour is a dict containing a list of order pairs for each 2d slice
    mapping can be label_id -> idx_in_cycle or None
    layer is a string in layerdict
    """
    # create a copy of image
    imagestack = np.copy(input_image)
    imagestack = np.tile(imagestack[:,:,:,np.newaxis], (1,1,1,3))

    if layer in layerdict.keys():
        layers = layerdict[layer]
    elif layer is not None:
        raise KeyError('layer {} is not recognized.'.format(layer))
    else:
        layers = [1,2,4]

    output_image = np.zeros(np.shape(imagestack))

    # overlay colours are fixed for a constant number of possible compartments
    if mapping is None:
        # uniques = np.sort(np.unique(segmentation.ravel()))
        uniques = [0,1,2,4]
        mapping = {i: c for c, i in enumerate(uniques)} 


    image = imagestack
    # rescale image to [0, 255]
    image = image - image.min()
    image = image / image.max() * 255

    for l in mapping.keys():
        if l==0:
            continue # skipping background here

        # do mask overlay
        if l==4 or contour is None:

            if l in layers:
                overlay = overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))
                if overlay_intensity == 1.0:
                    if len(layers) == 1:
                        # image[segmentation >= l] = overlay
                        image[segmentation & l == l] = overlay
                    else:
                        if l == 1:
                            image[segmentation & l == l] = overlay
                        else:
                            image[segmentation & l == l] = overlay
                else:
                    if len(layers) == 1:
                        # image[segmentation >= l] += overlay
                        image[segmentation & l == l] += overlay
                    else:
                        image[segmentation & l == l] += overlay
                    
        # do contour overlay. not finished yet or may be discarded even
        elif contour is not None:

            if l in layers:
                if layer in ['all','both']:
                    layerkey = layersdict[l]
                else:
                    layerkey = layer
                overlay = overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))
                for s in [s for s in contour[layerkey].keys() if len(contour[layerkey][s])>0]:
                    pairs = zip(contour[layerkey][s][0][:-2],contour[layerkey][s][0][1:])
                    for p1,p2 in pairs:
                        rr,cc,val = line_aa(int(p1[0]),int(p1[1]),int(p2[0]),int(p2[1]))
                        if overlay_intensity == 1.0:
                            image[int(s),rr,cc] = (val*np.reshape(overlay,(-1,1))).T
                        else:
                            image[int(s),rr,cc] += (val*np.reshape(overlay,(-1,1))).T


    # rescale result to [0,1]
    image = image / image.max() * 1
    # return image.astype(np.uint8)
    output_image = image

    output_image = np.squeeze(output_image)
    return output_image.astype(np.float32)

# a convenience method for the custom colormaps
def get_cmap(colormap):
    if colormap == 'tempo':
        return ListedColormap(np.array([[0 ,1, 0, 1],[0, .5, 0, 1]]))
    elif colormap == 'radnec':
        return ListedColormap(np.array([[0,1,1,1],[1,0,1,1]]))
    else:
        return None

# create overlays for z-score, cbv, tempo 
# mostly for the Overlay version of the viewer                
def generate_overlay(image: np.ndarray, overlay: np.ndarray = None, mask: np.ndarray = None, 
                     image_wl: np.ndarray = None, overlay_wl: np.ndarray = None,
                     overlay_intensity: float = 1.0, colormap: str = 'viridis'):
    """
    image,overlay is 3d grayscale
    mask is generally a unet or blast segmentation
    image_wl is the current window/level for the usual uint16 grayscale image data as currently displayed
    overlay_wl is the desired range for the overlay.
    """

    # create a copy of image
    image = np.copy(image)

    if mask is None:
        # general case for un-masked z-score or cbv
        mask_ros = np.where(overlay)
    else:
        mask_ros = np.where( (mask != 0) & (overlay != 0) )

    if overlay is not None:
        overlay = np.copy(overlay)
        # rescale overlay to provided window/level
        if overlay_wl is not None:
            overlay = overlay - (overlay_wl[1]-overlay_wl[0]/2)
            overlay = overlay / (overlay_wl[0])
            if False:
                overlay = np.clip(overlay,0,1)
        else:
            overlay = overlay - overlay.min()
            overlay = overlay / overlay.max() * 1
    else:
        overlay = np.zeros_like(image)

    if len(image.shape) == 3:
        image = np.tile(image[:,:,:,np.newaxis], (1,1,1,4))
    else:
        raise RuntimeError("unexpected image shape. only 3d grayscale images are supported")

    # rescale image data according to current window/level to [0, 1]
    if image_wl is not None:
        image = image - (image_wl[1]-image_wl[0]/2)
        image = image / (image_wl[0])
        image = np.clip(image,0,1)
    else:
        image = image - image.min()
        image = image / image.max() * 1
    image[:,:,:,3] = 1


    if colormap in plt.colormaps():
        cmap = plt.get_cmap(colormap)
    else:
        cmap = get_cmap(colormap)

    overlay_cmap = cmap(overlay)
    overlay_cmap[:,:,:,3] = overlay_intensity

    if overlay_intensity == 1:
        image[mask_ros] = overlay_cmap[mask_ros]
    else:
        # temp kludge. use window/level params.
        image[mask_ros] *= (1-overlay_intensity)
        image[mask_ros] += overlay_cmap[mask_ros]
        image = np.clip(image,a_min=0,a_max=1)
                     
    return image.astype(np.float32)

# composite overlay of BLAST plus SAM plus clicked points

def generate_comp_overlay(input_image: np.ndarray, overlay: np.ndarray = None, slice: int = 0, layer: str = None,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d or 3d grayscale, with a multi-channel dimension prepended.
    Segmentation must be label map of same shape as 2d or 3d image (w/o color channels)
    """
    # create a copy of image
    image = np.copy(input_image)
    image = np.tile(image[:,:,:,np.newaxis], (1,1,1,3))

    # rescale image to [0, 1]
    image = image - image.min()
    image = image / image.max() * 1

    uniques = [0,1,5,6,8] # these values are currently hard-coded in create_comp_mask
    mapping = {i: c for c, i in enumerate(uniques)} 
    # for now just using 1 for WT 
    overlay_color = overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[1]]))
    overlay_color /= 255
    # currently want to exclude raw BLAST pixels that are separate from the SAM mask
    overlay_ros = np.where((overlay>0) & (overlay != 5))

    alpha = np.zeros_like(overlay,dtype='float32')
    alpha = np.tile(alpha[:,:,:,np.newaxis],(1,1,1,3))
    # SAM only pixels
    alpha[overlay==1] = overlay_color * 1.0
    # BLAST only pixels
    # for now, not showing any raw BLAST pixel that is separate from the SAM roi.
    if False:
        alpha[overlay==5] = overlay_color * 0.6
    # SAM+BLAST pixels
    alpha[overlay==6] = overlay_color * 1.0

    # clicked point
    if True: # small circle
        clickedpts = np.where(overlay == 8)
        dim_z,dim_y,dim_x = np.shape(input_image)
        x = np.arange(0,dim_x)
        y = np.arange(0,dim_y)
        z = np.arange(0,dim_z)
        cmask = np.zeros_like(input_image,dtype='uint8')
        for p in zip(clickedpts[0],clickedpts[1],clickedpts[2]):
            # create grid
            vol = np.array(np.meshgrid(z,y,x,indexing='ij'))
            cmask1 = np.copy(cmask)
            cmask1[np.where(np.sqrt(np.power((vol[0,:]-p[0]),2)+np.power((vol[1,:]-p[1]),2)+np.power((vol[2,:]-p[2]),2)) < 4.5)] = 1
            cmask2 = np.copy(cmask)
            cmask2[np.where(np.sqrt(np.power((vol[0,:]-p[0]),2)+np.power((vol[1,:]-p[1]),2)+np.power((vol[2,:]-p[2]),2)) > 3.5)] = 1
            cmask = (cmask1) & (cmask2)
            alpha[cmask == 1] = overlay_color * 0.7
        alpha[overlay == 8] = overlay_color * 1.0
    else: # single point
        # alpha[overlay==8] = np.array(hex_to_rgb('69F570'))/255.
        alpha[overlay==8] = np.array(hex_to_rgb('000000'))/255.

    output_image = np.copy(image)
    output_image[overlay_ros] = alpha[overlay_ros]
    if False:
        plt.clf()
        plt.imshow(output_image[slice])

    # rescale result to [0,1]
    output_image = output_image / output_image.max() * 1
    output_image = np.squeeze(output_image)

    return output_image.astype(np.float32)


