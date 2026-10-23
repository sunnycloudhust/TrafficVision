# TrafficVision

TrafficVision trains and evaluates DETRAC vehicle detectors and runs four
tracking-by-detection pipelines. `main.py` is the command-line entry point.

## Models

- `faster_rcnn`: TorchVision Faster R-CNN ResNet-50-FPN with an RPN.
- `fast_rcnn`: shared image features and RoI Align with Selective Search boxes.
- `rcnn`: independent ResNet crop features for Selective Search boxes.

Fast R-CNN and R-CNN need `opencv-contrib-python` for `cv2.ximgproc` Selective
Search. Region proposals are cached under `outputs/selective_search_cache/`.
R-CNN is substantially slower because it runs the CNN separately on each box.

## Setup

Install a PyTorch and TorchVision version pair that matches your operating
system and accelerator using the official PyTorch installation selector. Then
install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The DETRAC dataset should be under `dataset/DETRAC/`. Training uses the 60
annotated train sequences and holds out whole sequences for validation. The 40
test sequences are reserved for final evaluation.

## Commands

List the supported detector names:

```bash
python main.py models
```

Train a detector. `--model` may be `faster_rcnn`, `fast_rcnn`, or `rcnn`.
Initialization can be `pretrained` or `scratch`.

```bash
python main.py train --model faster_rcnn --initialization pretrained
python main.py train --model fast_rcnn --initialization pretrained
python main.py train --model rcnn --initialization pretrained
```

Run training and DETRAC Test evaluation in sequence:

```bash
python main.py pipeline --model fast_rcnn --initialization pretrained
```

Evaluate a previously trained model. The model name selects its default
checkpoint; `--checkpoint` can point to a different one.

```bash
python main.py evaluate --model faster_rcnn
python main.py evaluate --model fast_rcnn
python main.py evaluate --checkpoint outputs/detrac_rcnn/best.pth --model rcnn
```

Options such as epochs, batch size, device, validation split, score thresholds,
and paths are in `configs/train.yaml`, `configs/test.yaml`, and
`configs/model.yaml`. Command-line values override the corresponding config
values. Training checkpoints are stored in model-specific directories under
`outputs/`; evaluation writes per-sequence detection CSVs and `metrics.json`.

## Tracking

First evaluate a trained detector on DETRAC Test. This exports per-sequence
detection CSV files to `outputs/evaluation_<model>/detections/`. Then run one
tracker against those detections:

```bash
python main.py track --tracker sort --detections-dir outputs/evaluation_faster_rcnn/detections
python main.py track --tracker bytetrack --detections-dir outputs/evaluation_faster_rcnn/detections
python main.py track --tracker botsort --detections-dir outputs/evaluation_faster_rcnn/detections
python main.py track --tracker ocsort --detections-dir outputs/evaluation_faster_rcnn/detections
python main.py track-all --detections-dir outputs/evaluation_faster_rcnn/detections
```

The selected tracker can also be set in `configs/tracking.yaml`. Tracker
settings include maximum lost age, minimum hits, IoU threshold and ByteTrack
confidence thresholds. BoT-SORT uses optical-flow camera-motion compensation
and color histograms as a lightweight appearance cue; it does not load a
learned ReID model. Results are saved as per-sequence track CSV files and
`<tracker>/` subfolders under `outputs/tracking/`, so running multiple
algorithms does not overwrite their CSVs. `track-all` also writes a
`comparison.json`. Metrics include project-implemented HOTA, IDF1, MOTA and ID
switches, reported per sequence and as an unweighted sequence macro average.
DETRAC Test annotation IDs are used as ground truth.

## Known gaps before benchmark or production use

- The tracker implementations are lightweight project versions, not exact
  reproductions of the authors' reference implementations. SORT uses a simple
  constant-velocity box model, OC-SORT does not implement every observation
  re-update step, and BoT-SORT uses color histograms instead of learned ReID.
- Tracking HOTA/IDF1/MOTA are implemented locally and should be cross-checked
  with the official TrackEval toolkit before reporting benchmark results.
- Training selects `best.pth` by validation loss. Validation AP/mAP-based
  checkpoint selection, resume-training, early stopping and a detailed epoch
  history are not implemented yet.
- There is no automated test suite or tracking visualization/export to video.
- Install a PyTorch/TorchVision pair compatible with your platform separately;
  `requirements.txt` lists the remaining packages. Put the DETRAC files under
  the documented `dataset/DETRAC/` layout; datasets and model outputs are
  excluded from Git by `.gitignore`.

These gaps do not block running the current train → detection evaluation →
tracking flow, but the tracker approximations and local metric implementation
limit direct comparison with published benchmark numbers.
