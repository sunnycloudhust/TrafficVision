"""R-CNN principle: crop raw proposals, run a CNN independently for every crop.

Uses a joint softmax + bbox head for end-to-end teaching experiments, rather than
the original paper's separate SVM training stages. This distinction is recorded
in run metadata/docs; compute is substantially higher than shared Fast R-CNN.
"""

from torch import nn
from torchvision.ops import roi_align
from ..fast_rcnn.model import FastRCNN


class ImageFeatures(nn.Module):
    def forward(self, images):
        return {"0": images}


class CropCNN(nn.Module):
    def __init__(self, backbone, crop_size=64):
        super().__init__()
        self.backbone = backbone
        self.crop_size = crop_size

    def forward(self, features, boxes, image_shapes):
        crops = roi_align(features["0"], boxes, self.crop_size, spatial_scale=1.0, aligned=True)
        if len(crops) == 0:
            return crops.new_zeros((0, self.backbone.out_channels, 7, 7))
        result = self.backbone(crops)
        result = result["0"] if isinstance(result, dict) else result
        return nn.functional.adaptive_avg_pool2d(result, (7, 7))


class RCNN(FastRCNN):
    def __init__(self, backbone="conv", init="random", channels=64, **kwargs):
        super().__init__(backbone=backbone, init=init, channels=channels, **kwargs)
        crop_backbone = self.backbone
        self.backbone = ImageFeatures()
        self.roi_heads.box_roi_pool = CropCNN(crop_backbone)
