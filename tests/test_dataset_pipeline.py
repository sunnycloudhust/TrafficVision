import json
from trafficvision.pipeline.dataset_pipeline import track_dataset
from trafficvision.pipeline.detection_tracking import DetectionTrackingPipeline
from trafficvision.smoke import SyntheticDetector
from trafficvision.registry import build_tracker
from trafficvision.evaluation.tracking_metrics import tracking_metrics


def test_all_sequences_export_with_matching_video_ids(prepared, tmp_path):
    pipeline = DetectionTrackingPipeline(SyntheticDetector(), build_tracker("sort", dict(min_hits=1)))
    report = track_dataset(
        dict(data=dict(source="local", manifest=str(prepared / "frames.jsonl"))),
        "test",
        pipeline,
        tmp_path / "tracks.csv",
    )
    assert report["sequences"] == 1 and report["frames"] == 1
    gt = json.loads((tmp_path / "tracks.gt.jsonl").read_text())
    assert gt["video_id"] == "MVI_2"
    metrics = tracking_metrics(report["ground_truth"], report["predictions"])
    assert metrics["sequences"] == 1
