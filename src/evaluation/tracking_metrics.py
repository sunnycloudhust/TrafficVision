"""TrackEval adapter and JSON summary extraction for MOT-format sequence exports."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np


def run_trackeval(trackeval_root, gt_root, tracker_root, seqmap_file, output_dir,
                  benchmark: str, split: str, tracker_name: str = "FasterRCNN-ByteTrack") -> dict:
    trackeval_root = Path(trackeval_root)
    if not (trackeval_root / "scripts" / "run_mot_challenge.py").is_file():
        raise FileNotFoundError(f"TrackEval checkout not found at {trackeval_root}")
    sys.path.insert(0, str(trackeval_root))
    # Older TrackEval source uses the NumPy alias removed after NumPy 1.23.
    if not hasattr(np, "float"):
        np.float = float
    import trackeval

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_CONFIG": False, "PRINT_RESULTS": True,
                     "TIME_PROGRESS": False, "OUTPUT_FOLDER": str(output_dir), "PLOT_CURVES": False})
    dataset_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_cfg.update({"GT_FOLDER": str(Path(gt_root)), "TRACKERS_FOLDER": str(Path(tracker_root)),
                        "OUTPUT_FOLDER": str(output_dir), "TRACKERS_TO_EVAL": [tracker_name],
                        "BENCHMARK": benchmark, "SPLIT_TO_EVAL": split, "SEQMAP_FILE": str(Path(seqmap_file)),
                        "CLASSES_TO_EVAL": ["pedestrian"], "DO_PREPROC": False})
    metric_cfg = {"THRESHOLD": .5, "PRINT_CONFIG": False}
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(metric_cfg), trackeval.metrics.Identity()]
    result, messages = evaluator.evaluate([dataset], metrics)
    status = next(iter(messages.values()))
    if any(message != "Success" for message in status.values()):
        raise RuntimeError(f"TrackEval did not complete successfully: {messages}")
    dataset_result = next(iter(result.values()))[tracker_name]

    def extract(values):
        return {"HOTA": float(values["HOTA"]["HOTA"].mean()),
                "IDF1": float(values["Identity"]["IDF1"]), "MOTA": float(values["CLEAR"]["MOTA"])}

    combined = extract(dataset_result["COMBINED_SEQ"]["pedestrian"])
    combined.update({"per_sequence": {seq: extract(value["pedestrian"]) for seq, value in dataset_result.items()
                                      if seq != "COMBINED_SEQ"},
                     "aggregation": "TrackEval combined-sequence metrics; vehicle classes treated as one foreground class",
                     "split": split, "tracker": tracker_name})
    (output_dir / "tracking_metrics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined
