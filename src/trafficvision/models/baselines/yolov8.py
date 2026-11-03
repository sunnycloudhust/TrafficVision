"""YOLO baseline mapping is explicit: COCO has no UA-DETRAC van class."""

from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm


class YOLOv8Detector:
    def __init__(
        self, weights="yolov8n.pt", device="cpu", confidence=0.05, label_space="auto", image_size=640
    ):
        from ultralytics import YOLO

        if label_space not in ("auto", "coco", "ua_detrac"):
            raise ValueError("YOLO label_space must be auto, coco or ua_detrac")
        self.model = YOLO(str(weights))
        names = self.model.names
        canonical = ["car", "bus", "van", "others"]
        is_ua_detrac = len(names) == 4 and [names.get(i) for i in range(4)] == canonical
        is_coco = all(names.get(i) == name for i, name in {2: "car", 5: "bus", 7: "truck"}.items())
        if label_space == "auto":
            label_space = "ua_detrac" if is_ua_detrac else "coco" if is_coco else "unknown"
        if (
            (label_space == "ua_detrac" and not is_ua_detrac)
            or (label_space == "coco" and not is_coco)
            or label_space == "unknown"
        ):
            raise ValueError(
                "Checkpoint class names differ from configured label_space; verify class mapping"
            )
        self.label_space = label_space
        self.device, self.confidence, self.image_size = device, confidence, image_size
        self.mapping = {2: 1, 5: 2, 7: 4} if label_space == "coco" else {0: 1, 1: 2, 2: 3, 3: 4}

    def predict(self, image):
        rgb = (image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        result = self.model.predict(
            rgb[:, :, ::-1].copy(),
            device=self.device,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )[0].boxes.cpu()
        labels = result.cls.long()
        keep = torch.tensor([int(v) in self.mapping for v in labels], dtype=torch.bool)
        return dict(
            boxes=result.xyxy[keep],
            scores=result.conf[keep],
            labels=torch.tensor([self.mapping[int(v)] for v in labels[keep]], dtype=torch.int64),
        )


def export_yolo_dataset(data_config, destination):
    """Materialize the same canonical splits to runtime, including streamed shards."""
    from ...data.datasets.ua_detrac import build_dataset
    from torchvision.transforms.functional import to_pil_image
    from ...data.transforms import mask_ignored_pixels
    import yaml

    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("YOLO export requires an empty destination to avoid mixing old dataset splits")
    for split in ("train", "val", "test"):
        images_dir, labels_dir = destination / "images" / split, destination / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        dataset = build_dataset(data_config, split)
        try:
            total = len(dataset)
        except (TypeError, NotImplementedError):
            total = None
        for image, target in tqdm(dataset, total=total, desc=f"Exporting YOLO {split}", unit="image"):
            key = f"{int(target['image_id']):09d}"
            to_pil_image(mask_ignored_pixels(image, target)).save(images_dir / f"{key}.jpg")
            height, width = image.shape[-2:]
            lines = []
            for box, label in zip(target["boxes"], target["labels"]):
                x1, y1, x2, y2 = box.tolist()
                lines.append(
                    f"{int(label)-1} {(x1+x2)/2/width} {(y1+y2)/2/height} {(x2-x1)/width} {(y2-y1)/height}"
                )
            (labels_dir / f"{key}.txt").write_text("\n".join(lines))
    path = destination / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                path=str(destination),
                train="images/train",
                val="images/val",
                test="images/test",
                names={0: "car", 1: "bus", 2: "van", 3: "others"},
            )
        )
    )
    return path
