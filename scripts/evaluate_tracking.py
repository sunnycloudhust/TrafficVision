"""Run ByteTrack on held-out UA-DETRAC sequences and evaluate with TrackEval."""
from __future__ import annotations
import argparse, configparser, json
from pathlib import Path
import sys
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import build_dataset
from src.detection.predictor import Predictor
from src.models.loading import load_model_checkpoint
from src.evaluation.tracking_metrics import run_trackeval
from src.tracking.bytetrack import ByteTrackAdapter
from src.utils.config import choose_device, load_yaml


def ignored_regions(xml_path: Path):
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    return [[float(b.attrib["left"]), float(b.attrib["top"]),
             float(b.attrib["left"]) + float(b.attrib["width"]),
             float(b.attrib["top"]) + float(b.attrib["height"])]
            for b in root.findall("./ignored_region/box")]


def overlaps_ignore(box, regions, threshold=.5):
    x1, y1, x2, y2 = box
    area = max(0, x2-x1) * max(0, y2-y1)
    for a1, b1, a2, b2 in regions:
        ix, iy = max(x1,a1), max(y1,b1); jx, jy = min(x2,a2), min(y2,b2)
        inter = max(0,jx-ix)*max(0,jy-iy)
        if area and inter / area >= threshold: return True
    return False


def mot_line(frame, ident, box, score, class_id=1):
    x1,y1,x2,y2 = map(float, box)
    return f"{frame},{ident},{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},{score:.6f},{class_id},1\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--tracking-config", default="configs/tracking.yaml")
    parser.add_argument("--trackeval-root")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args()
    eval_cfg = load_yaml(args.config); trk_cfg = load_yaml(args.tracking_config)
    root = Path(args.config).resolve().parent.parent
    ds_cfg = load_yaml(root / eval_cfg["dataset_config"])
    if ds_cfg.get("format") != "detrac_xml":
        raise ValueError("This tracking exporter currently requires UA-DETRAC XML with per-frame target IDs")
    model, state = load_model_checkpoint(args.checkpoint, choose_device(eval_cfg.get("device", "auto")))
    if state["model_config"]["class_names"] != ds_cfg["class_names"]:
        raise ValueError("Checkpoint class mapping does not match the configured dataset")
    dataset = build_dataset(ds_cfg, args.split, training=False)
    ann_dir = Path(ds_cfg["test_annotations"] if args.split == "test" else ds_cfg["train_annotations"])
    xml_by_seq = {p.stem: p for p in ann_dir.glob("*.xml")}
    out_root = root / trk_cfg.get("mot_output_dir", "outputs/tracking") / Path(args.checkpoint).parent.name
    benchmark = f"DETRAC-{args.split}"
    gt_root = out_root / "gt" / benchmark; tracker_root = out_root / "trackers" / benchmark / "FasterRCNN-ByteTrack" / "data"
    seqmap = out_root / "gt" / "seqmaps" / f"{benchmark}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "".join(f"{seq}\n" for seq in dataset.sequences), encoding="utf-8")
    predictor = Predictor(model, choose_device(eval_cfg.get("device", "auto")), float(trk_cfg.get("score_threshold", .05)))
    tracker = None
    by_sequence = {seq: [] for seq in dataset.sequences}
    for record in dataset.records:
        by_sequence[record["sequence"]].append(record)
    for sequence, records in by_sequence.items():
        records.sort(key=lambda r: r["frame"])
        xml_path = xml_by_seq[sequence]
        ignore = ignored_regions(xml_path)
        tracker = ByteTrackAdapter({**trk_cfg, "frame_rate": int(trk_cfg.get("frame_rate", 30))})
        gt_path = gt_root / sequence / "gt" / "gt.txt"; gt_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path = tracker_root / f"{sequence}.txt"; pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_lines = []
        image_root = Path(ds_cfg["image_root"])
        first = cv2.imread(str(image_root / sequence / f"img{records[0]['frame']:05d}.jpg"))
        if first is None: raise FileNotFoundError(f"Missing first frame for {sequence}")
        height,width = first.shape[:2]
        with gt_path.open("w",encoding="utf-8") as gt_stream:
            for record in records:
                for box, ident in zip(record["boxes"], record["track_ids"]):
                    gt_stream.write(mot_line(record["frame"], int(ident), box, 1.0))
                image = cv2.imread(str(image_root / sequence / f"img{record['frame']:05d}.jpg"))
                if image is None: raise FileNotFoundError(f"Missing frame: {sequence}/{record['frame']}")
                detections = predictor.predict_bgr(image)
                tracked = tracker.update(detections)
                for item in tracked:
                    if not overlaps_ignore(item.box, ignore):
                        pred_lines.append(mot_line(record["frame"], item.track_id, item.box, item.score))
        pred_path.write_text("".join(pred_lines), encoding="utf-8")
        seqinfo = configparser.ConfigParser()
        seqinfo["Sequence"] = {"name": sequence, "imDir": "img1", "frameRate": str(trk_cfg.get("frame_rate", 30)),
                                "seqLength": str(max(r["frame"] for r in records)), "imWidth": str(width), "imHeight": str(height), "imExt": ".jpg"}
        with (gt_root / sequence / "seqinfo.ini").open("w", encoding="utf-8") as f: seqinfo.write(f)
    trackeval_root = Path(args.trackeval_root or eval_cfg.get("trackeval_root") or "")
    output = out_root / "metrics"; output.mkdir(parents=True, exist_ok=True)
    summary = run_trackeval(trackeval_root, out_root / "gt", out_root / "trackers", seqmap,
                            output, "DETRAC", args.split)
    summary["ignored_region_prediction_rule"] = "discard prediction if >=50% of its area overlaps a UA-DETRAC ignored box"
    (output / "tracking_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"per_sequence"}}, indent=2))
    print(f"MOT-format GT/predictions and TrackEval reports saved under {out_root}")


if __name__ == "__main__": main()
