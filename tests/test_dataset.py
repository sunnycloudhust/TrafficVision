import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

from src.datasets.vehicle_dataset import VehicleDataset, detection_collate


def test_detrac_boxes_labels_and_empty_frame(tmp_path):
    image_root = tmp_path / "images"
    (image_root / "SEQ_1").mkdir(parents=True)
    for number in (1, 2):
        assert cv2.imwrite(str(image_root / "SEQ_1" / f"img{number:05d}.jpg"), np.zeros((24, 32, 3), np.uint8))
    root = ET.Element("sequence", name="SEQ_1")
    frame = ET.SubElement(root, "frame", num="1")
    targets = ET.SubElement(frame, "target_list")
    target = ET.SubElement(targets, "target", id="7")
    ET.SubElement(target, "box", left="2", top="3", width="10", height="8")
    ET.SubElement(target, "attribute", vehicle_type="car")
    ET.SubElement(root, "frame", num="2")
    ann = tmp_path / "SEQ_1.xml"
    ET.ElementTree(root).write(ann)

    dataset = VehicleDataset(image_root, ann, ["car"])
    image, target = dataset[0]
    assert image.shape == (3, 24, 32)
    assert image.dtype == torch.float32 and image.min() >= 0 and image.max() <= 1
    assert torch.equal(target["boxes"], torch.tensor([[2., 3., 12., 11.]]))
    assert torch.equal(target["labels"], torch.tensor([1]))
    assert torch.equal(target["track_ids"], torch.tensor([7]))
    assert dataset[1][1]["boxes"].shape == (0, 4)
    images, targets = detection_collate([dataset[0], dataset[1]])
    assert len(images) == len(targets) == 2
