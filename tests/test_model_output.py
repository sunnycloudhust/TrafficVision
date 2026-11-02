import torch
import pytest
from trafficvision.registry import build_model

torch.set_num_threads(2)


@pytest.mark.parametrize("name", ["faster_rcnn", "fast_rcnn", "rcnn", "rtdetr"])
@pytest.mark.parametrize("empty", [False, True])
def test_training_backward_and_original_coordinates(name, empty):
    params = (
        dict(image_size=64, queries=10, decoder_layers=2)
        if name == "rtdetr"
        else dict(min_size=64, max_size=64, proposals=20)
    )
    if name in ("fast_rcnn", "rcnn"):
        params["proposal_strategy"] = "grid"
    torch.manual_seed(7)
    model = build_model(name, **params)
    image = torch.rand(3, 48, 64)
    target = dict(
        boxes=torch.empty((0, 4)) if empty else torch.tensor([[8.0, 5.0, 40.0, 30.0]]),
        labels=torch.empty(0, dtype=torch.long) if empty else torch.tensor([1]),
    )
    model.train()
    losses = model([image], [target])
    loss = sum(losses.values())
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        (output,) = model([image])
    assert set(output) == {"boxes", "labels", "scores"}
    assert output["boxes"].shape == (len(output["scores"]), 4)
    if len(output["boxes"]):
        assert output["boxes"].min() >= 0
        assert output["boxes"][:, 0::2].max() <= 64
        assert output["boxes"][:, 1::2].max() <= 48
        assert output["labels"].min() >= 1 and output["labels"].max() <= 4
