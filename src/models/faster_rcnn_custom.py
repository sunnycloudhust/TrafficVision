"""Faster R-CNN ResNet-50-FPN initialized entirely at random."""
from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_scratch_model(config: dict, seed: int = 42):
    """Create a randomly initialized detector; no backbone weights are downloaded."""
    names = config["class_names"]
    num_classes = len(names) + 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None,
            box_score_thresh=float(config.get("score_threshold", 0.05)),
            box_nms_thresh=float(config.get("nms_threshold", 0.5)),
            box_detections_per_img=int(config.get("box_detections_per_img", 100)),
            min_size=int(config.get("min_size", 800)), max_size=int(config.get("max_size", 1333)),
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
