"""Geometry-preserving utilities shared by dataset export tools."""

import torch


def mask_ignored_pixels(image, target):
    """Mask ignored areas for YOLO, whose stock loss has no ignore-box interface.

    Preserve labeled positives even when they overlap an ignored rectangle.
    Evaluation still uses original images and the common prediction ignore rule.
    """
    height, width = image.shape[-2:]
    mask = torch.zeros((height, width), dtype=torch.bool, device=image.device)
    for region in target.get("ignored_regions", []):
        x1, y1, x2, y2 = region.floor().long().tolist()
        mask[max(0, y1) : max(0, min(height, y2)), max(0, x1) : max(0, min(width, x2))] = True
    for region in target["boxes"]:
        x1, y1 = region[:2].floor().long().tolist()
        x2, y2 = region[2:].ceil().long().tolist()
        mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = False
    image = image.clone()
    image[:, mask] = 0
    return image
