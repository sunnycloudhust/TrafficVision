import torch

from src.models.faster_rcnn_custom import build_scratch_model
from src.training.checkpoint import load_checkpoint, save_checkpoint


def test_cpu_inference_shapes_and_checkpoint(tmp_path):
    config = {"class_names": ["car"], "score_threshold": 0.0, "nms_threshold": 0.5}
    model = build_scratch_model(config, seed=3).cpu().eval()
    model.transform.min_size = (64,)
    model.transform.max_size = 64
    with torch.inference_mode():
        prediction = model([torch.rand(3, 64, 64)])[0]
    assert prediction["boxes"].ndim == 2 and prediction["boxes"].shape[1] == 4
    assert prediction["scores"].ndim == 1 and prediction["labels"].ndim == 1
    assert len(prediction["boxes"]) == len(prediction["scores"]) == len(prediction["labels"])
    assert torch.all(prediction["scores"] >= 0) and torch.all(prediction["scores"] <= 1)
    assert torch.all(prediction["boxes"][:, 0::2] >= 0) and torch.all(prediction["boxes"][:, 0::2] <= 64)
    assert torch.all(prediction["boxes"][:, 1::2] >= 0) and torch.all(prediction["boxes"][:, 1::2] <= 64)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, 2, 0.25, config, {"1": "car"}, 3)
    clone = build_scratch_model(config, seed=4)
    state = load_checkpoint(path, clone)
    assert state["epoch"] == 2
    for expected, actual in zip(model.parameters(), clone.parameters()):
        assert torch.equal(expected, actual)
