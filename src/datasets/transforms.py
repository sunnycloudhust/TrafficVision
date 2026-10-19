"""Image and target transforms for Faster R-CNN."""
from __future__ import annotations

import random
import torch


def horizontal_flip(image: torch.Tensor, target: dict, probability: float = 0.5):
    """Flip an RGB CHW image and pixel-coordinate boxes together."""
    if random.random() >= probability:
        return image, target
    width = image.shape[-1]
    image = image.flip(-1)
    target = dict(target)
    boxes = target["boxes"].clone()
    if boxes.numel():
        old_left = boxes[:, 0].clone()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - old_left
    target["boxes"] = boxes
    return image, target


def to_image_tensor(rgb):
    """Convert an RGB uint8 NumPy array into float32 CHW in [0, 1]."""
    import numpy as np
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image; got {rgb.shape}")
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255)
