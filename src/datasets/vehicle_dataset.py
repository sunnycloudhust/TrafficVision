"""UA-DETRAC XML and COCO JSON detection datasets."""
from __future__ import annotations

from pathlib import Path
import json
import math
import random
import xml.etree.ElementTree as ET
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset

from .transforms import horizontal_flip, to_image_tensor


def detection_collate(batch):
    return tuple(zip(*batch))


class VehicleDataset(Dataset):
    """Frame-level vehicle records with contiguous labels starting at 1."""

    def __init__(self, image_root: str | Path, annotation_path: str | Path,
                 class_names: list[str], annotation_format: str = "detrac_xml",
                 category_mapping: dict | None = None, sequence_names: set[str] | None = None,
                 training: bool = False, flip_probability: float = 0.5):
        self.image_root = Path(image_root)
        self.annotation_path = Path(annotation_path)
        self.class_names = list(class_names)
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must be a non-empty list of unique names")
        self.label_by_name = {name: index + 1 for index, name in enumerate(self.class_names)}
        self.category_mapping = category_mapping or {}
        self.training = training
        self.flip_probability = flip_probability
        if annotation_format == "detrac_xml":
            self.records = self._read_detrac(sequence_names)
        elif annotation_format == "coco":
            self._sequence_names = sequence_names
            self.records = self._read_coco()
        else:
            raise ValueError("annotation_format must be 'detrac_xml' or 'coco'")
        if not self.records:
            raise ValueError(f"No image records found in {self.annotation_path}")

    def _read_detrac(self, sequence_names: set[str] | None) -> list[dict[str, Any]]:
        paths = [self.annotation_path] if self.annotation_path.is_file() else sorted(self.annotation_path.glob("*.xml"))
        if not paths:
            raise FileNotFoundError(f"No XML annotation files found under {self.annotation_path}")
        records = []
        for xml_path in paths:
            root = ET.parse(xml_path).getroot()
            sequence = root.attrib.get("name", xml_path.stem)
            if sequence_names is not None and sequence not in sequence_names:
                continue
            ignored = []
            for region_box in root.findall("./ignored_region/box"):
                ix, iy = float(region_box.attrib["left"]), float(region_box.attrib["top"])
                iw, ih = float(region_box.attrib["width"]), float(region_box.attrib["height"])
                ignored.append([ix, iy, ix + iw, iy + ih])
            for frame in root.findall("frame"):
                frame_num = int(frame.attrib["num"])
                boxes, labels, ids, areas = [], [], [], []
                for target in frame.findall("./target_list/target"):
                    box = target.find("box")
                    attr = target.find("attribute")
                    if box is None or attr is None:
                        continue
                    source_name = attr.attrib.get("vehicle_type")
                    class_name = self.category_mapping.get(source_name, source_name)
                    if class_name not in self.label_by_name:
                        continue
                    x, y = float(box.attrib["left"]), float(box.attrib["top"])
                    w, h = float(box.attrib["width"]), float(box.attrib["height"])
                    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError(f"Invalid box in {xml_path}, frame {frame_num}: {(x,y,w,h)}")
                    boxes.append([x, y, x + w, y + h])
                    labels.append(self.label_by_name[class_name])
                    ids.append(int(target.attrib["id"]))
                    areas.append(w * h)
                image_path = self.image_root / sequence / f"img{frame_num:05d}.jpg"
                records.append({"path": image_path, "sequence": sequence, "frame": frame_num,
                                "boxes": boxes, "labels": labels, "track_ids": ids, "areas": areas,
                                "ignore_boxes": ignored})
        return records

    def _read_coco(self) -> list[dict[str, Any]]:
        data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        categories = {c["id"]: c["name"] for c in data["categories"]}
        mapped = {}
        for category_id, name in categories.items():
            normalized = self.category_mapping.get(str(category_id), self.category_mapping.get(name, name))
            if normalized in self.label_by_name:
                mapped[category_id] = self.label_by_name[normalized]
        by_image: dict[int, list] = {image["id"]: [] for image in data["images"]}
        ignored_by_image: dict[int, list] = {image["id"]: [] for image in data["images"]}
        for ann in data["annotations"]:
            x, y, w, h = map(float, ann["bbox"])
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"Invalid COCO bbox in annotation {ann.get('id')}")
            if ann.get("iscrowd", 0):
                ignored_by_image[ann["image_id"]].append((x, y, x+w, y+h))
                continue
            if ann["category_id"] not in mapped:
                continue
            by_image[ann["image_id"]].append((x, y, x+w, y+h, mapped[ann["category_id"]], ann.get("id", -1), w*h))
        records = []
        for image in data["images"]:
            sequence = str(image.get("video_id", image["id"]))
            if self._sequence_names is not None and sequence not in self._sequence_names:
                continue
            values = by_image[image["id"]]
            records.append({"path": self.image_root / image["file_name"], "sequence": sequence,
                            "frame": int(image["id"]), "boxes": [v[:4] for v in values],
                            "labels": [v[4] for v in values], "track_ids": [v[5] for v in values],
                            "areas": [v[6] for v in values], "ignore_boxes": ignored_by_image[image["id"]]})
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(record["path"]))
        if image is None:
            raise FileNotFoundError(f"Cannot read image {record['path']}")
        image = to_image_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        if boxes.numel():
            h, w = image.shape[-2:]
            boxes[:, 0].clamp_(0, w); boxes[:, 2].clamp_(0, w)
            boxes[:, 1].clamp_(0, h); boxes[:, 3].clamp_(0, h)
            if not torch.isfinite(boxes).all() or torch.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError(f"Invalid or out-of-frame box for {record['path']}")
        labels = torch.as_tensor(record["labels"], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels,
                  "image_id": torch.tensor(index, dtype=torch.int64),
                  "area": torch.as_tensor(record["areas"], dtype=torch.float32),
                  "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
                  "track_ids": torch.as_tensor(record["track_ids"], dtype=torch.int64),
                  "ignore_boxes": torch.as_tensor(record.get("ignore_boxes", []), dtype=torch.float32).reshape(-1, 4),
                  "sequence": record["sequence"], "frame": record["frame"]}
        if self.training:
            image, target = horizontal_flip(image, target, self.flip_probability)
        return image, target

    @property
    def sequences(self) -> list[str]:
        return sorted({r["sequence"] for r in self.records})


def build_dataset(config: dict, split: str, training: bool = False) -> VehicleDataset:
    """Build a split; DETRAC validation is selected by sequence, never by frame."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    fmt = config.get("format", "detrac_xml")
    if fmt not in {"detrac_xml", "coco"}:
        raise ValueError("format must be detrac_xml or coco")
    fraction = float(config.get("validation_fraction", .15))
    if split != "test" and not 0 < fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    classes = config["class_names"]
    mapping = config.get("category_mapping", {})
    if fmt == "detrac_xml":
        if split == "test":
            ann_path = config["test_annotations"]
            names = None
        else:
            ann_path = config["train_annotations"]
            all_files = sorted(Path(ann_path).glob("*.xml"))
            names_all = [ET.parse(p).getroot().attrib.get("name", p.stem) for p in all_files]
            rng = random.Random(int(config.get("split_seed", 42)))
            rng.shuffle(names_all)
            n_val = max(1, round(len(names_all) * float(config.get("validation_fraction", 0.15))))
            val_names = set(names_all[:n_val])
            names = val_names if split == "val" else set(names_all) - val_names
    else:
        ann_path = config["test_annotations"] if split == "test" else config["train_annotations"]
        if split == "test":
            names = None
        else:
            data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
            groups = sorted({str(i.get("video_id", i["id"])) for i in data["images"]})
            rng = random.Random(int(config.get("split_seed", 42)))
            rng.shuffle(groups)
            n_val = max(1, round(len(groups) * float(config.get("validation_fraction", 0.15))))
            val_names = set(groups[:n_val])
            names = val_names if split == "val" else set(groups) - val_names
    return VehicleDataset(config["image_root"], ann_path, classes, fmt, mapping, names, training)
