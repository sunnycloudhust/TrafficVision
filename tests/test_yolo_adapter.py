from types import SimpleNamespace
import torch
import pytest
from trafficvision.models.baselines.yolov8 import YOLOv8Detector, export_yolo_dataset


def test_pretrained_mapping_and_wrong_checkpoint(monkeypatch):
    names = {0: "person", 2: "car", 5: "bus", 7: "truck"}
    fake = SimpleNamespace(names=names)
    monkeypatch.setattr("ultralytics.YOLO", lambda weights: fake)
    adapter = YOLOv8Detector("unused.pt")
    assert adapter.label_space == "coco" and adapter.mapping == {2: 1, 5: 2, 7: 4}
    with pytest.raises(ValueError, match="class names"):
        YOLOv8Detector("unused.pt", label_space="ua_detrac")


def test_export_preserves_split_and_label_offset(prepared, tmp_path):
    path = export_yolo_dataset(
        dict(source="local", manifest=str(prepared / "frames.jsonl")), tmp_path / "yolo"
    )
    assert path.exists()
    labels = list((path.parent / "labels/train").glob("*.txt"))
    assert len(labels) == 1 and labels[0].read_text().split()[0] == "0"
    with pytest.raises(FileExistsError, match="old dataset splits"):
        export_yolo_dataset(dict(source="local", manifest=str(prepared / "frames.jsonl")), path.parent)


def test_selective_search_actual_forward():
    from trafficvision.registry import build_model

    model = build_model("fast_rcnn", min_size=64, max_size=64, proposals=10).eval()
    with torch.inference_mode():
        (result,) = model([torch.rand(3, 64, 64)])
    assert result["boxes"].shape[1] == 4
