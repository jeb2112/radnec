from typing import Tuple, Union
import numpy as np
import pandas as pd

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


def generate_overlay(input_image: np.ndarray, segmentation: np.ndarray, mapping: dict = None,
                     color_cycle: Tuple[str, ...] = color_cycle,
                     overlay_intensity: float = 0.6):
    """
    image can be 2d or 3d grayscale, with a multi-channel dimension prepended.

    Segmentation must be label map of same shape as 2d or 3d image (w/o color channels)

    mapping can be label_id -> idx_in_cycle or None

    """
    # create a copy of image
    imagestack = np.copy(input_image)

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

    for ch in range(n_channel):

        image = imagestack[ch]
        # rescale image to [0, 255]
        image = image - image.min()
        image = image / image.max() * 255

        # create output
        if mapping is None:
            uniques = np.sort(pd.unique(segmentation.ravel()))  # np.unique(segmentation)
            mapping = {i: c for c, i in enumerate(uniques)} 

        for l in mapping.keys():
            if l==0:
                continue # skipping background here
            overlay = overlay_intensity * np.array(hex_to_rgb(color_cycle[mapping[l]]))
            image[segmentation == l] += overlay

        # rescale result to [0,1]
        image = image / image.max() * 1
        # return image.astype(np.uint8)
        output_image[ch] = image

    output_image = np.squeeze(output_image)
    return output_image.astype(np.float32)






    # overlay_pred = generate_overlay(image[selected_slice], pred[selected_slice], overlay_intensity=overlay_intensity)
 

