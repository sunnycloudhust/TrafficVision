"""Small scratch CNN for classroom experiments; no implicit weight downloads."""

from collections import OrderedDict
import torch.nn as nn


def block(inputs, outputs, stride=1):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(),
        nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(),
    )


class ConvBackbone(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        if channels % 8:
            raise ValueError("channels must be a multiple of 8")
        self.body = nn.Sequential(block(3, 32, 2), block(32, channels, 2), block(channels, channels, 2))
        self.out_channels = channels

    def forward(self, images):
        return OrderedDict([("0", self.body(images))])


def build_backbone(name="conv", init="random", channels=64):
    if init not in ("random", "imagenet"):
        raise ValueError("Custom detector init must be random or imagenet")
    if name == "conv":
        if init != "random":
            raise ValueError("The custom ConvBackbone has no ImageNet weights; choose resnet18")
        return ConvBackbone(channels)
    if name == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        model = resnet18(weights=ResNet18_Weights.DEFAULT if init == "imagenet" else None)
        backbone = nn.Sequential(*list(model.children())[:-2])
        backbone.out_channels = 512
        return backbone
    raise ValueError(f"Unknown backbone {name}")
