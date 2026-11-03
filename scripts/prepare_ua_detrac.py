"""Parse original XML and frames, write local JSONL, reproducible splits and TAR shards."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

# Allow direct execution from a checkout that uses the src/ package layout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficvision.data.parsers.xml_parser import parse_sequence, CLASS_MAP  # noqa: E402
from trafficvision.data.splits import split_sequences, validate_splits
from trafficvision.data.cache import sha256_file


def prepare(images, annotations, output, splits_path=None, seed=42, shard_size=500, ratios=(0.7, 0.15, 0.15)):
    images, annotations, output = Path(images), Path(annotations), Path(output)
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    files = sorted(annotations.rglob("*.xml"))
    if not files:
        raise ValueError("No XML annotation files found")
    sequences = {}
    for path in files:
        rows = parse_sequence(path)
        if not rows:
            raise ValueError(f"No frames in {path}")
        name = rows[0]["video_id"]
        if name in sequences:
            raise ValueError(f"Duplicate annotation for sequence {name}")
        sequences[name] = rows
    splits = (
        json.loads(Path(splits_path).read_text()) if splits_path else split_sequences(sequences, seed, ratios)
    )
    validate_splits(splits)
    if set(splits) != {"train", "val", "test"} or set(sum(splits.values(), [])) != set(sequences):
        raise ValueError("Splits must cover exactly all annotated sequences using train/val/test")
    # Discover images instead of assuming an archive directory layout.
    image_paths = {}
    for path in images.rglob("*"):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        video = next((p.name for p in path.parents if p.name in sequences), None)
        digits = path.stem.removeprefix("img")
        if video and digits.isdigit():
            key = (video, int(digits))
            if key in image_paths:
                raise ValueError(f"Ambiguous image for {key}")
            image_paths[key] = path.resolve()
    grouped = defaultdict(list)
    image_id = 0
    for split, videos in splits.items():
        for video in videos:
            for row in sequences[video]:
                key = (video, row["frame_id"])
                if key not in image_paths:
                    raise FileNotFoundError(f"Missing image {video}, frame {row['frame_id']}")
                row = dict(row, split=split, image_id=image_id)
                grouped[split].append((image_paths[key], row))
                image_id += 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    shards = []
    with (output / "frames.jsonl").open("w") as manifest:
        for split, samples in grouped.items():
            for path, row in samples:
                manifest.write(json.dumps(dict(row, image=os.path.relpath(path, output.resolve()))) + "\n")
            for start in range(0, len(samples), shard_size):
                shard_path = output / f"{split}-{start // shard_size:05d}.tar"
                with tarfile.open(shard_path, "w") as tar:
                    for path, row in samples[start : start + shard_size]:
                        if path.suffix.lower() in (".jpg", ".jpeg"):
                            # Preserve source JPEGs; decoding and recompressing every frame is slow
                            # and can introduce avoidable image differences.
                            payload = path.read_bytes()
                        else:
                            from PIL import Image

                            with Image.open(path) as image:
                                buffer = io.BytesIO()
                                image.convert("RGB").save(buffer, format="JPEG", quality=95)
                            payload = buffer.getvalue()
                        row = dict(row, image_sha256=hashlib.sha256(payload).hexdigest())
                        key = f"{row['video_id']}_{row['frame_id']:06d}"
                        for ext, content in (("jpg", payload), ("json", json.dumps(row).encode())):
                            member = tarfile.TarInfo(f"{key}.{ext}")
                            member.size = len(content)
                            tar.addfile(member, io.BytesIO(content))
                shards.append(
                    dict(
                        url=shard_path.name,
                        split=split,
                        sha256=sha256_file(shard_path),
                        size=shard_path.stat().st_size,
                        samples=len(samples[start : start + shard_size]),
                    )
                )
    metadata = dict(
        schema_version=1,
        class_map=CLASS_MAP,
        seed=seed,
        splits=splits,
        preferred_mode="cache",
        shards=shards,
        counts={k: len(v) for k, v in grouped.items()},
        coordinates="continuous xyxy; original XML origin retained",
    )
    (output / "shards.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-size", type=int, default=500)
    args = p.parse_args()
    print(
        json.dumps(
            prepare(args.images, args.annotations, args.output, args.splits, args.seed, args.shard_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
