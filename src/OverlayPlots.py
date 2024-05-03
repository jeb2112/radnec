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


def generate_overlay(image: np.ndarray, overlay: np.ndarray = None, image_wl: np.ndarray = None, overlay_wl: np.ndarray = None,
                     overlay_intensity: float = 0.5, colormap: str = 'viridis'):
    """
    image,overlay is 3d grayscale

    overlay should also be pre-masked 
    image_wl is the current window/level for the usual uint16 grayscale image data as currently displayed
    overlay_wl is the desired range for the overlay.
    """

    # create a copy of image
    image = np.copy(image)
    overlay = np.copy(overlay)
    overlay_mask = np.where(overlay > 0)


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

    if overlay_wl is not None:
        overlay = overlay - (overlay_wl[1]-overlay_wl[0]/2)
        overlay = overlay / (overlay_wl[0])
    else:
        overlay = overlay - overlay.min()
        overlay = overlay / overlay.max() * 1

    cmap = plt.get_cmap(colormap)

    overlay = cmap(overlay)
    overlay[:,:,:,3] = overlay_intensity
    image[overlay_mask] = overlay[overlay_mask]
                     
    return image.astype(np.float32)

