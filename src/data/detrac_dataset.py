"""PyTorch dataset for the DETRAC image and XML annotation layout."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset


DEFAULT_CLASS_NAMES = ("car", "van", "bus", "others")
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "DETRAC"


class DETRACDataset(Dataset):
    """Read DETRAC frames and return targets in torchvision detection format.

    Foreground labels are one-based in ``class_names`` order; label zero is
    reserved for Faster R-CNN's background class. Each target also contains
    ``track_ids`` so frame-level predictions can later be matched to DETRAC
    identities during tracking evaluation.

    Args:
        root: DETRAC directory containing the image and annotation folders.
        split: ``"train"`` or ``"test"``.
        class_names: Ordered foreground class names used for label IDs.
        sequences: Optional sequence-name allowlist. Sequence ordering is
            preserved, and frames within each sequence are ordered by XML frame
            number.
        include_empty: Include frames that contain no targets.
    """

    _SPLIT_DIRS = {
        "train": "DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML",
        "test": "DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML",
    }
    _IMAGE_DIR = "DETRAC-Images/DETRAC-Images"

    def __init__(
        self,
        root: str | Path = DEFAULT_DATASET_ROOT,
        split: str = "train",
        class_names: tuple[str, ...] | list[str] = DEFAULT_CLASS_NAMES,
        sequences: list[str] | tuple[str, ...] | None = None,
        include_empty: bool = True,
    ) -> None:
        if split not in self._SPLIT_DIRS:
            raise ValueError("split must be 'train' or 'test'")
        if not class_names or len(class_names) != len(set(class_names)):
            raise ValueError("class_names must contain unique foreground classes")

        self.root = Path(root)
        self.split = split
        self.class_names = tuple(class_names)
        self.label_by_name = {name: idx + 1 for idx, name in enumerate(self.class_names)}
        self.include_empty = include_empty
        self.image_root = self.root / self._IMAGE_DIR
        annotation_root = self.root / self._SPLIT_DIRS[split]
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"DETRAC image directory not found: {self.image_root}")
        if not annotation_root.is_dir():
            raise FileNotFoundError(f"DETRAC {split} annotation directory not found: {annotation_root}")

        xml_paths = {path.stem: path for path in annotation_root.glob("*.xml")}
        sequence_names = list(sequences) if sequences is not None else sorted(xml_paths)
        missing_xml = [name for name in sequence_names if name not in xml_paths]
        if missing_xml:
            raise FileNotFoundError(f"No {split} XML annotation for sequence(s): {missing_xml}")

        self.records: list[dict] = []
        for sequence_name in sequence_names:
            image_dir = self.image_root / sequence_name
            if not image_dir.is_dir():
                raise FileNotFoundError(f"DETRAC image sequence not found: {image_dir}")
            sequence = ET.parse(xml_paths[sequence_name]).getroot()
            if sequence.tag != "sequence":
                raise ValueError(f"Expected <sequence> root in {xml_paths[sequence_name]}")

            for frame in sequence.findall("frame"):
                frame_number = int(frame.attrib["num"])
                image_path = image_dir / f"img{frame_number:05d}.jpg"
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image for {sequence_name} frame {frame_number} not found: {image_path}"
                    )

                boxes: list[list[float]] = []
                labels: list[int] = []
                track_ids: list[int] = []
                for target in frame.findall("./target_list/target"):
                    attribute = target.find("attribute")
                    vehicle_type = attribute.attrib.get("vehicle_type") if attribute is not None else None
                    if vehicle_type not in self.label_by_name:
                        raise ValueError(
                            f"Unknown or missing vehicle_type {vehicle_type!r} in "
                            f"{sequence_name} frame {frame_number}; configured classes: {self.class_names}"
                        )
                    box = target.find("box")
                    if box is None:
                        raise ValueError(f"Target without <box> in {sequence_name} frame {frame_number}")
                    left = float(box.attrib["left"])
                    top = float(box.attrib["top"])
                    width = float(box.attrib["width"])
                    height = float(box.attrib["height"])
                    if width <= 0 or height <= 0:
                        continue
                    boxes.append([left, top, left + width, top + height])
                    labels.append(self.label_by_name[vehicle_type])
                    track_ids.append(int(target.attrib["id"]))

                if self.include_empty or boxes:
                    self.records.append({
                        "image_path": image_path,
                        "sequence": sequence_name,
                        "frame_number": frame_number,
                        "boxes": boxes,
                        "labels": labels,
                        "track_ids": track_ids,
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image_tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)

        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(record["labels"], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "track_ids": torch.tensor(record["track_ids"], dtype=torch.int64),
            "image_path": str(record["image_path"]),
        }
        return image_tensor, target

    def get_frame_info(self, index: int) -> dict[str, str | int]:
        """Return sequence and frame metadata for an item without loading pixels."""
        record = self.records[index]
        return {"sequence": record["sequence"], "frame_number": record["frame_number"]}


def detection_collate_fn(batch):
    """Keep variable-size image and target tensors as lists for torchvision."""
    return tuple(zip(*batch))
