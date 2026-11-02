"""Explicit Faster R-CNN assembly from reusable torchvision operators.

The backbone, anchors, RPN, RoI pooling, MLP and prediction heads are independently
configured here. This is not a renamed pretrained detection model.
"""

import torch
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from ...backbones.convnet import build_backbone
from .ignore import IgnoreRPN, IgnoreRoIHeads, IgnoreTransform


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes=4,
        init="random",
        backbone="conv",
        channels=64,
        min_size=512,
        max_size=960,
        proposals=300,
        detections_per_image=100,
        score_threshold=0.05,
        **kwargs,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, init, channels)
        channels = self.backbone.out_channels
        anchors = AnchorGenerator(((16, 32, 64, 128, 256),), ((0.5, 1.0, 2.0),))
        self.rpn = IgnoreRPN(
            anchors,
            RPNHead(channels, anchors.num_anchors_per_location()[0]),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=128,
            positive_fraction=0.5,
            pre_nms_top_n={"training": proposals * 2, "testing": proposals * 2},
            post_nms_top_n={"training": proposals, "testing": proposals},
            nms_thresh=0.7,
        )
        self.roi_heads = IgnoreRoIHeads(
            MultiScaleRoIAlign(["0"], 7, 2),
            TwoMLPHead(channels * 7 * 7, 256),
            FastRCNNPredictor(256, num_classes + 1),
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=128,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=score_threshold,
            nms_thresh=0.5,
            detections_per_img=detections_per_image,
        )
        self.transform = IgnoreTransform(
            min_size, max_size, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Training requires detection targets")
        original_sizes = [image.shape[-2:] for image in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, rpn_losses = self.rpn(images, features, targets)
        detections, roi_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_sizes)
        return {**rpn_losses, **roi_losses} if self.training else detections
