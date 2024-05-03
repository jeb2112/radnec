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


def hex_to_rgb(hex: str):
    assert len(hex) == 6
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def generate_overlay(image: np.ndarray, overlay: np.ndarray = None, 
                     overlay_intensity: float = 0.6, colormap: str = 'viridis'):
    """
    image,overlay is 3d grayscale

    overlay is a masked image of same shape as 2d or 3d image

    """

    # create a copy of image
    image = np.copy(image)
    overlay = np.copy(overlay)
    overlay_mask = np.where(overlay > 0)


    if len(image.shape) == 3:
        image = np.tile(image[:,:,:,np.newaxis], (1,1,1,4))
    else:
        raise RuntimeError("unexpected image shape. only 3d grayscale images are supported")
    # if len(overlay.shape) == 3:
    #     overlay = np.tile(overlay[:,:,:,np.newaxis], (1,1,1,3))
    # else:
    #     raise RuntimeError("unexpected overlay shape. only 3d grayscale images are supported")

    # rescale image to [0, 1]
    image = image - image.min()
    image = image / image.max() * 1
    overlay = overlay - overlay.min()
    overlay = overlay / overlay.max() * 1

    cmap = plt.get_cmap('jet')

    overlay = cmap(overlay)
    overlay[:,:,:,0] = overlay_intensity
    image[overlay_mask] = overlay[overlay_mask]
                    
    # rescale result to [0,1]
    image = image / image.max() * 1
 
    return image.astype(np.float32)

