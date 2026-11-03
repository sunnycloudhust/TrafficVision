import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms.functional import pil_to_tensor
from ..sources.local import LocalDatasetSource
from ..sources.cloud_storage import CloudObjectStorageSource
from ..sources.huggingface_stream import HuggingFaceStreamingSource


def convert_sample(image, record):
    boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
    width, height = image.size
    boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2].clamp_(0, height)
    keep = (boxes[:, 2:] > boxes[:, :2]).all(1)
    boxes = boxes[keep]
    target = {
        "boxes": boxes,
        "labels": torch.tensor(record["labels"], dtype=torch.int64)[keep],
        "track_ids": torch.tensor(record["track_ids"], dtype=torch.int64)[keep],
        "image_id": torch.tensor(int(record["image_id"])),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        "ignored_regions": torch.tensor(record.get("ignored_regions", []), dtype=torch.float32).reshape(
            -1, 4
        ),
    }
    return pil_to_tensor(image).float() / 255, target


class UADETRACDataset(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return convert_sample(*self.source[index])


class StreamingUADETRACDataset(IterableDataset):
    def __init__(self, source):
        self.source = source

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        for image, row in self.source:
            yield convert_sample(image, row)


def build_dataset(config, split):
    kind = config.get("source", "local")
    if kind == "local":
        return UADETRACDataset(LocalDatasetSource(config["manifest"], split))
    sources = {"cloud": CloudObjectStorageSource, "huggingface": HuggingFaceStreamingSource}
    if kind not in sources:
        raise ValueError(f"Unknown dataset source: {kind}")
    return StreamingUADETRACDataset(
        sources[kind](
            config["manifest"],
            split,
            mode=config.get("mode", "auto"),
            cache_dir=config.get("cache_dir"),
            seed=config.get("seed", 42),
            shuffle=split == "train",
        )
    )


def collate_detection(batch):
    return tuple(zip(*batch))
