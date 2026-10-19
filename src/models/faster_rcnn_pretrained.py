"""COCO-pretrained Faster R-CNN adapted to the configured vehicle classes."""
from __future__ import annotations

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_pretrained_model(config: dict):
    """Retain COCO backbone/RPN/box feature weights; replace the classifier head."""
    weights_name = config.get("pretrained_weights", "DEFAULT")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if weights_name == "DEFAULT" else getattr(FasterRCNN_ResNet50_FPN_Weights, weights_name)
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        box_score_thresh=float(config.get("score_threshold", 0.05)),
        box_nms_thresh=float(config.get("nms_threshold", 0.5)),
        box_detections_per_img=int(config.get("box_detections_per_img", 100)),
        min_size=int(config.get("min_size", 800)), max_size=int(config.get("max_size", 1333)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(config["class_names"]) + 1)
    if config.get("freeze_backbone", False):
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    return model
