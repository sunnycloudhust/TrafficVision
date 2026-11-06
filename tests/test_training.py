import json
import torch
import pytest
from trafficvision.training.trainer import train
from trafficvision.registry import load_detector
from trafficvision.config import load_config


@pytest.mark.parametrize("model_name", ["faster_rcnn", "fast_rcnn", "rcnn", "rtdetr"])
def test_train_save_resume_evaluate(prepared, tmp_path, model_name):
    torch.set_num_threads(2)
    params = (
        dict(image_size=64, queries=10, decoder_layers=2)
        if model_name == "rtdetr"
        else dict(min_size=64, max_size=64, proposals=20)
    )
    if model_name in ("fast_rcnn", "rcnn"):
        params["proposal_strategy"] = "grid"
    config = dict(
        seed=7,
        data=dict(source="local", manifest=str(prepared / "frames.jsonl")),
        model=dict(name=model_name, init="random", params=params),
        training=dict(
            output=str(tmp_path / "train"), epochs=1, batch_size=1, workers=0, device="cpu", max_steps=1
        ),
    )
    output = train(config)
    assert (output / "best.pt").exists() and (output / "last.pt").exists()
    detector = load_detector(model_name, output / "best.pt")
    result = detector.predict(torch.rand(3, 48, 64))
    assert result["boxes"].shape[1] == 4
    config["training"]["epochs"] = 2
    train(config, resume=output / "last.pt")
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in records] == [0, 1]
    assert "map_50" in records[-1]["metrics"]


def test_config_inheritance():
    config = load_config("configs/experiments/faster_rcnn_scratch.yaml")
    base = load_config("configs/experiments/base.yaml")
    assert config["model"]["name"] == "faster_rcnn"
    assert config["training"]["lr"] == base["training"]["lr"]
