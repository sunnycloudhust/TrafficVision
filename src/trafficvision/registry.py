"""Lazy factories keep dataset preparation independent of optional model backends."""


def build_model(name, init="random", **params):
    from .models.detectors.faster_rcnn.model import FasterRCNN
    from .models.detectors.fast_rcnn.model import FastRCNN
    from .models.detectors.rcnn.model import RCNN
    from .models.detectors.rtdetr.model import RTDETR

    constructors = {"faster_rcnn": FasterRCNN, "fast_rcnn": FastRCNN, "rcnn": RCNN, "rtdetr": RTDETR}
    if name not in constructors:
        raise ValueError(f"Unknown custom detector: {name}. YOLO is a separate baseline.")
    return constructors[name](init=init, **params)


def load_detector(name, checkpoint=None, config=None, device="cpu", confidence=0.05):
    import torch
    from .models.base import TorchDetector

    if name == "yolov8":
        from .models.baselines.yolov8 import YOLOv8Detector

        params = (config or {}).get("model", {})
        return YOLOv8Detector(
            checkpoint or params.get("weights", "yolov8n.pt"),
            device,
            confidence,
            label_space=params.get("label_space", "auto"),
            image_size=params.get("image_size", 640),
        )
    if checkpoint is None:
        raise ValueError("Custom inference requires a trained checkpoint; use smoke for random-weight tests")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["model_name"] != name:
        raise ValueError("Checkpoint architecture does not match requested detector")
    # A checkpoint is self-describing; init random avoids any download when restoring.
    model = build_model(name, init="random", **saved["model_params"])
    model.load_state_dict(saved["model_state"])
    return TorchDetector(model, device, confidence)


def build_tracker(name, config=None, fps=25):
    if name == "sort":
        from .tracking.sort import SORT

        return SORT(**(config or {}))
    if name in ("bytetrack", "botsort"):
        from .tracking.bytetrack import UltralyticsTracker

        return UltralyticsTracker(name, config or {}, fps)
    raise ValueError(f"Unknown tracker: {name}")
