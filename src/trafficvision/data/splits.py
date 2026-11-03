"""Deterministic video-disjoint splits; persist the actual sequence lists."""

import random


def split_sequences(video_ids, seed=42, ratios=(0.7, 0.15, 0.15)):
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("Three nonnegative split ratios must sum to one")
    videos = sorted(set(video_ids))
    if len(videos) < 3:
        raise ValueError("At least three sequences required for train/val/test")
    random.Random(seed).shuffle(videos)
    a, b = int(len(videos) * ratios[0]), int(len(videos) * ratios[1])
    if a == 0 or b == 0 or a + b == len(videos):
        raise ValueError("Empty split; supply more videos or adjust split ratios")
    return dict(train=sorted(videos[:a]), val=sorted(videos[a : a + b]), test=sorted(videos[a + b :]))


def validate_splits(splits):
    all_ids = [v for values in splits.values() for v in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Video leakage between splits or duplicate sequences")
