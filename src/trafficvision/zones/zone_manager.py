import torch
from torchvision.ops import batched_nms
from .polygon_zone import PolygonZone


class ZoneManager:
    def __init__(self, config=None):
        config = config or {}
        self.zones = [PolygonZone(name, **params) for name, params in config.get("zones", {}).items()]
        self.frame_size = config.get("frame_size")
        self.camera_id = config.get("camera_id")
        self.roi_only = config.get("roi_only", True)
        self.default_threshold = config.get("default_threshold", 0.25)

    def validate_frame(self, width, height):
        if self.frame_size and list(self.frame_size) != [width, height]:
            raise ValueError("Zone coordinates do not match video frame size; recalibrate the config")

    def assignments(self, box):
        return [zone for zone in self.zones if zone.contains_box(box)]

    def filter(self, detections):
        keep = []
        for box, score in zip(detections["boxes"], detections["scores"]):
            zones = self.assignments(box)
            threshold = min((z.confidence_threshold for z in zones), default=self.default_threshold)
            keep.append(bool(score >= threshold and (zones or not self.roi_only or not self.zones)))
        mask = torch.tensor(keep, dtype=torch.bool)
        return {key: value[mask] for key, value in detections.items()}

    def detect(self, detector, image):
        """Extra crop passes increase far-region pixels; then class-aware global NMS."""
        height, width = image.shape[-2:]
        self.validate_frame(width, height)
        predictions = [detector.predict(image)]
        for zone in self.zones:
            if zone.inference_scale <= 1:
                continue
            low, high = zone.polygon.min(0), zone.polygon.max(0)
            x1, y1 = max(0, int(low[0])), max(0, int(low[1]))
            x2, y2 = min(width, int(high[0]) + 1), min(height, int(high[1]) + 1)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Zone {zone.name} falls outside frame")
            crop = image[:, y1:y2, x1:x2]
            scaled = torch.nn.functional.interpolate(
                crop[None], scale_factor=zone.inference_scale, mode="bilinear", align_corners=False
            )[0]
            prediction = detector.predict(scaled)
            # Use actual rounded resize dimensions, not nominal scale.
            scale = prediction["boxes"].new_tensor(
                [scaled.shape[-1] / (x2 - x1), scaled.shape[-2] / (y2 - y1)] * 2
            )
            prediction["boxes"] = prediction["boxes"] / scale + prediction["boxes"].new_tensor(
                [x1, y1, x1, y1]
            )
            mask = torch.tensor([zone.contains_box(b) for b in prediction["boxes"]], dtype=torch.bool)
            predictions.append({key: value[mask] for key, value in prediction.items()})
        combined = {key: torch.cat([p[key] for p in predictions]) for key in ("boxes", "scores", "labels")}
        keep = batched_nms(combined["boxes"], combined["scores"], combined["labels"], 0.5)
        return self.filter({key: value[keep] for key, value in combined.items()})
