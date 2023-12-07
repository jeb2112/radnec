from typing import Tuple, Union
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from skimage.draw import line_aa

color_cycle = (
    "000000",
    "4363d8",
    "f58231",
    "3cb44b",
    "e6194B",
    "911eb4",
    "ffe119",
    "bfef45",
    "42d4f4",
    "f032e6",
    "000075",
    "9A6324",
    "808000",
    "800000",
    "469990",
)

# hard-coded convention
# layerdict = {'ET':[3],'TC':[2],'WT':[1],'T2 hyper':[1],'all':[1,2,3],'both':[1,3]}
layerdict = {'ET':[4],'TC':[2],'WT':[1],'T2 hyper':[1],'all':[1,2,4],'both':[1,4]}
# need both integer indexing for the mask overlay and dict keyword indexing for contour overlay
# this is a convenience definition for reverse lookup from integer back to dict keyword, for
# 'all' and 'both' in contour overlay, but it's awkward
# layersdict = {1:'WT',2:'TC',3:'ET'}
layersdict = {1:'WT',2:'TC',4:'ET'}

def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray = None, contour: dict = None, layer: str = None,
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

    if layer in layerdict.keys():
        layers = layerdict[layer]
    elif layer is not None:
        raise KeyError('layer {} is not recognized.'.format(layer))
    else:
        layers = [1,2,4]

    if len(imagestack.shape) == 4:
        if imagestack.shape[0]<5:
            imagestack = np.tile(imagestack[:,:,:,:,np.newaxis],(1,1,1,1,3))
        else:
            raise RuntimeError(f'if 3d grayscale image is given the first dimension must be the channel. '
                               f'Your image shape: {imagestack.shape}')
    else:
        if len(imagestack.shape) == 3:
            if imagestack.shape[0] < 5: 
                imagestack = np.tile(imagestack[:,:,:,np.newaxis], (1,1,1,3))
            else:
                raise RuntimeError(f'if 2d grayscale image is given the first dimension must be the channel. '
                                f'Your image shape: {imagestack.shape}')
        else:
            raise RuntimeError("unexpected image shape. only 2d and 3d grayscale images are supported")
    n_channel = imagestack.shape[0]
    output_image = np.zeros(np.shape(imagestack))

    # overlay colours are fixed for a constant number of possible compartments
    if mapping is None:
        # uniques = np.sort(np.unique(segmentation.ravel()))
        uniques = [0,1,2,4]
        mapping = {i: c for c, i in enumerate(uniques)} 

    for ch in range(n_channel):

        image = imagestack[ch]
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
                        
            # do contour overlay. not finished yet.
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
        output_image[ch] = image

    output_image = np.squeeze(output_image)
    return output_image.astype(np.float32)

def select_layer(segmentation,layer):
    layersegmentation = copy.deepcopy(segmentation)
    layersegmentation = (segmentation == layerdict[layer]).astype('int')*layerdict[layer]
    return layersegmentation

 

