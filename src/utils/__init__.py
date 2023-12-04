from __future__ import absolute_import

from .imutils import *
from .misc import *
from .osutils import *
from .transforms import *

import torch.nn.functional as F

def trimTensor(tensor, target_shape):
    # Find which dimensions need to be trimmed
    h_diff = tensor.shape[2] - target_shape[2]
    w_diff = tensor.shape[3] - target_shape[3]

    # Return the tensor if it is already the correct shape
    if h_diff == 0 and w_diff == 0:
        return tensor

    # Find the starting and ending indices for each dimension
    h_start = h_diff // 2
    h_end = h_start + target_shape[2]
    w_start = w_diff // 2
    w_end = w_start + target_shape[3]

    # Trim the tensor
    cropped_tensor = tensor[:, :, h_start:h_end, w_start:w_end]
    return cropped_tensor

def padTensor(tensor, target_height, target_width, mode='constant'):
    # Find which dimensions need to be padded
    h_diff = target_height - tensor.shape[2]
    w_diff = target_width - tensor.shape[3]

    # Return the tensor if it is already the correct shape
    if h_diff == 0 and w_diff == 0:
        return tensor

    # Calculate padding for each dimension
    pad_height = h_diff // 2
    pad_width = w_diff // 2

    # Pad the tensor
    padded_tensor = F.pad(tensor, (pad_width, pad_width, pad_height, pad_height), mode=mode)

    return padded_tensor

def trimNPArray(npArray, target_shape):
    # Find which dimensions need to be trimmed
    h_diff = npArray.shape[0] - target_shape[0]
    w_diff = npArray.shape[1] - target_shape[1]

    # Return the array if it is already the correct shape
    if h_diff == 0 and w_diff == 0:
        return npArray

    # Find the starting and ending indices for each dimension
    h_start = h_diff // 2
    h_end = h_start + target_shape[0]
    w_start = w_diff // 2
    w_end = w_start + target_shape[1]

    # Trim the array
    cropped_array = npArray[h_start:h_end, w_start:w_end]
    return cropped_array