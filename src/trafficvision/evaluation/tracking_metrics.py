"""TrackEval HOTA/CLEAR/Identity on canonical sequence records and demo CSV.

Uses actual trajectory IDs, class-gated IoU, dense per-sequence ID remapping,
and explicit frame alignment. This is a project protocol, not UA-DETRAC PR-MOTA.
"""

from collections import defaultdict
import csv
import json
import numpy as np
from tqdm.auto import tqdm
from ..tracking.sort import iou_matrix


def trackeval_backend():
    import trackeval
    from trackeval.metrics import hota, identity

    class NumpyCompatibility:
        # Pinned official TrackEval predates removal of np.float/np.int.
        # Give only those modules a view; never mutate the numpy module globally.
        float = float
        int = int

        def __getattr__(self, key):
            return getattr(np, key)

    hota.np = identity.np = NumpyCompatibility()
    return trackeval


def sequence_data(ground_truth, predictions):
    if not ground_truth:
        raise ValueError("Missing ground-truth frames")
    ground_truth = {int(row["frame_id"]): row for row in ground_truth}
    if set(predictions) - set(ground_truth):
        raise ValueError("Predictions contain frames absent from ground truth")
    gt_maps, pred_maps = {}, {}
    gt_ids, tracker_ids, similarities = [], [], []
    for frame_id in tqdm(
        range(1, max(ground_truth) + 1), desc="Aligning sequence", unit="frame", leave=False
    ):
        row = ground_truth.get(frame_id, dict(boxes=[], labels=[], track_ids=[], ignored_regions=[]))
        records = predictions.get(frame_id, [])
        boxes = np.array([r["box"] for r in records], dtype=float).reshape(-1, 4)
        labels = np.array([r["label"] for r in records], dtype=int)
        truth = np.array(row["boxes"], dtype=float).reshape(-1, 4)
        similarity = iou_matrix(truth, boxes)
        similarity *= np.array(row["labels"])[:, None] == labels[None]
        keep = np.ones(len(boxes), dtype=bool)
        for j, box in enumerate(boxes):
            center = (box[:2] + box[2:]) / 2
            ignored = any(
                np.all(center >= region[:2]) and np.all(center <= region[2:])
                for region in np.asarray(row.get("ignored_regions", []))
            )
            if ignored and not (similarity[:, j] >= 0.5).any():
                keep[j] = False
        records = [r for r, k in zip(records, keep) if k]
        tids = [int(r["track_id"]) for r in records]
        gids = list(map(int, row["track_ids"]))
        if len(tids) != len(set(tids)) or len(gids) != len(set(gids)):
            raise ValueError(f"Duplicate trajectory IDs in frame {frame_id}")
        for key in gids:
            gt_maps.setdefault(key, len(gt_maps))
        for key in tids:
            pred_maps.setdefault(key, len(pred_maps))
        gt_ids.append(np.array([gt_maps[v] for v in gids], dtype=int))
        tracker_ids.append(np.array([pred_maps[v] for v in tids], dtype=int))
        similarities.append(similarity[:, keep])
    return dict(
        num_gt_ids=len(gt_maps),
        num_tracker_ids=len(pred_maps),
        num_timesteps=len(gt_ids),
        num_gt_dets=sum(map(len, gt_ids)),
        num_tracker_dets=sum(map(len, tracker_ids)),
        gt_ids=gt_ids,
        tracker_ids=tracker_ids,
        similarity_scores=similarities,
    )


def tracking_metrics(gt_manifest, prediction_csv, split="test"):
    trackeval = trackeval_backend()
    gt, pred = defaultdict(list), defaultdict(lambda: defaultdict(list))
    with open(gt_manifest) as handle:
        for line in tqdm(handle, desc="Loading tracking ground truth", unit="record", leave=False):
            row = json.loads(line)
            if row["split"] == split:
                gt[row["video_id"]].append(row)
    with open(prediction_csv) as handle:
        predictions = tqdm(
            csv.DictReader(handle), desc="Loading tracking predictions", unit="record", leave=False
        )
        for row in predictions:
            if row["video_id"] not in gt:
                raise ValueError(f"Prediction video outside {split} split: {row['video_id']}")
            pred[row["video_id"]][int(row["frame_id"])].append(
                dict(
                    box=[float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                    label=int(row["label"]),
                    track_id=int(row["track_id"]),
                )
            )
    if not gt:
        raise ValueError(f"Empty ground-truth split: {split}")
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"PRINT_CONFIG": False}),
    ]
    results = {metric.get_name(): {} for metric in metrics}
    for video, rows in tqdm(gt.items(), total=len(gt), desc="TrackEval sequences", unit="sequence"):
        data = sequence_data(rows, pred[video])
        for metric in metrics:
            results[metric.get_name()][video] = metric.eval_sequence(data)
    combined = {metric.get_name(): metric.combine_sequences(results[metric.get_name()]) for metric in metrics}
    return dict(
        HOTA=float(np.mean(combined["HOTA"]["HOTA"])),
        IDF1=float(combined["Identity"]["IDF1"]),
        MOTA=float(combined["CLEAR"]["MOTA"]),
        ID_switches=int(combined["CLEAR"]["IDSW"]),
        fragments=int(combined["CLEAR"]["Frag"]),
        sequences=len(gt),
        split=split,
        protocol="TrackEval; class-aware IoU; bbox-center ignore; fractions [0,1]; not PR-MOTA",
    )
