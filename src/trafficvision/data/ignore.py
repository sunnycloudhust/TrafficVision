import torch


def centers_in_regions(boxes, regions):
    if len(boxes) == 0 or len(regions) == 0:
        return torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return (
        (centers[:, None] >= regions[None, :, :2]).all(-1)
        & (centers[:, None] <= regions[None, :, 2:]).all(-1)
    ).any(1)
