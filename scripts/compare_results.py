"""Write a CSV and Markdown comparison without inventing missing metrics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

FIELDS = ["Model", "Initialization", "Fine-tuned", "Precision", "Recall", "mAP@50", "HOTA", "IDF1", "MOTA"]

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None

def row(model, init, tuned, det=None, track=None):
    det = det or {}; track = track or {}
    if "metrics" in det: det = det["metrics"]
    return {"Model": model, "Initialization": init, "Fine-tuned": tuned,
            "Precision": det.get("precision", "N/A"), "Recall": det.get("recall", "N/A"),
            "mAP@50": det.get("map50", "N/A"), "HOTA": track.get("HOTA", track.get("hota", "N/A")),
            "IDF1": track.get("IDF1", track.get("idf1", "N/A")), "MOTA": track.get("MOTA", track.get("mota", "N/A"))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", required=True); p.add_argument("--fine-tuned", required=True)
    p.add_argument("--scratch-tracking"); p.add_argument("--fine-tuned-tracking")
    p.add_argument("--pretrained-only"); p.add_argument("--pretrained-only-tracking")
    p.add_argument("--output-dir", default="outputs/metrics")
    args = p.parse_args()
    rows = [row("Faster R-CNN", "Random", "No", load(args.scratch), load(args.scratch-tracking)),
            row("Faster R-CNN", "COCO pretrained", "Yes", load(args.fine_tuned), load(args.fine_tuned_tracking))]
    if args.pretrained_only:
        rows.insert(1, row("Faster R-CNN", "COCO pretrained", "No", load(args.pretrained_only), load(args.pretrained_only_tracking)))
    else:
        rows.insert(1, row("Faster R-CNN", "COCO pretrained", "No"))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    def fmt(v): return f"{v:.4f}" if isinstance(v, (int,float)) else str(v)
    md = ["| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    md += ["| " + " | ".join(fmt(r[k]) for k in FIELDS) + " |" for r in rows]
    (out / "comparison.md").write_text("\n".join(md) + "\n\nPretrained-only metrics require an explicit category mapping; unavailable values are shown as N/A. Tracking metrics require TrackEval and ground-truth IDs.\n", encoding="utf-8")
    print(f"Wrote {out / 'comparison.csv'} and {out / 'comparison.md'}")

if __name__ == "__main__": main()
