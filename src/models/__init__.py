"""Faster R-CNN model constructors."""
from .faster_rcnn_custom import build_scratch_model
from .faster_rcnn_pretrained import build_pretrained_model


def build_model(config: dict):
    names = config.get("class_names", [])
    if not names or len(names) != len(set(names)):
        raise ValueError("class_names must contain unique foreground class names")
    for key in ("score_threshold", "nms_threshold"):
        if key in config and not 0 <= float(config[key]) <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if int(config.get("min_size", 800)) < 1 or int(config.get("max_size", 1333)) < 1:
        raise ValueError("model image sizes must be positive")
    configured_count = config.get("num_classes")
    if configured_count is not None and int(configured_count) != len(names) + 1:
        raise ValueError("num_classes must include one background class in addition to class_names")
    #### Build models ### 
    init = config.get("initialization", "scratch")
    if init == "scratch":
        return build_scratch_model(config)
    if init == "pretrained":
        return build_pretrained_model(config)
    raise ValueError("initialization must be 'scratch' or 'pretrained'")


__all__ = ["build_model", "build_scratch_model", "build_pretrained_model"]
