import csv
import json
import pytest
from trafficvision.evaluation.tracking_metrics import tracking_metrics


def test_real_tracking_metrics_detect_identity_switch(tmp_path):
    truth = tmp_path / "frames.jsonl"
    truth.write_text(
        "\n".join(
            json.dumps(
                dict(
                    video_id="MVI",
                    frame_id=i,
                    split="test",
                    boxes=[[0, 0, 10, 10]],
                    labels=[1],
                    track_ids=[8],
                )
            )
            for i in range(1, 5)
        )
    )
    pred = tmp_path / "pred.csv"

    def write(switch):
        with pred.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["video_id", "frame_id", "track_id", "label", "x1", "y1", "x2", "y2"]
            )
            writer.writeheader()
            for i in range(1, 5):
                writer.writerow(
                    dict(
                        video_id="MVI",
                        frame_id=i,
                        track_id=2 if switch and i > 2 else 1,
                        label=1,
                        x1=0,
                        y1=0,
                        x2=10,
                        y2=10,
                    )
                )

    write(False)
    perfect = tracking_metrics(truth, pred)
    assert perfect["HOTA"] == pytest.approx(1) and perfect["IDF1"] == pytest.approx(1)
    write(True)
    switched = tracking_metrics(truth, pred)
    assert switched["ID_switches"] == 1 and switched["IDF1"] < 1 and switched["HOTA"] < 1
