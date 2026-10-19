# TrafficVision

Modular vehicle detection and tracking with Torchvision Faster R-CNN and ByteTrack. The project compares the same ResNet-50-FPN detector initialized randomly and initialized from COCO then fine-tuned. It also includes held-out detection evaluation, video inference, and TrackEval export for HOTA, IDF1, and MOTA.

## Dataset and labels

The UA-DETRAC root is read from [`data/dataset_path.txt`](data/dataset_path.txt); relative values are resolved from the `data/` folder. Expected image and annotation subfolders are listed in [`configs/dataset.yaml`](configs/dataset.yaml). The annotation files contain `vehicle_type`, pixel boxes (`left`, `top`, `width`, `height`), frame numbers, and per-sequence target IDs. All four observed labels are configured: `car`, `van`, `bus`, and `others`. Label 0 is reserved for Faster R-CNN background; the configured classes are assigned contiguous labels 1–4.

The official UA-DETRAC test sequences stay held out. The 60 official training sequences are split deterministically by video into train and validation (default seed 42, validation fraction 0.15), so frames from one video cannot cross those splits. The 40 official test sequences include ground-truth IDs, so tracking metrics can be calculated. Change paths, classes, and mappings in YAML for another dataset.

The alternate dataset adapter accepts COCO JSON with `images`, `annotations`, and `categories`; boxes use COCO `[x,y,width,height]`. For video data, set `video_id` on images so the validation split stays sequence-based. Give a distinct held-out test annotation file. `category_mapping` maps source class names or IDs to configured `class_names`. Dataset validation fails on missing images or invalid boxes. No class or path mapping should be inferred for a new dataset.

## Install

Use Python 3.12 for the Intel Mac environment represented in this workspace. PyTorch stopped publishing tested macOS x86_64 builds after 2.2; the compatible final pair is Torch 2.2.2 and Torchvision 0.17.2. The current system Python is 3.13, which is outside the macOS installer's recommended range. These versions are pinned in `requirements.txt`; see the [PyTorch macOS install guide](https://docs.pytorch.org/get-started/locally/) and [version pairing table](https://docs.pytorch.org/get-started/previous-versions/).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On this Intel Mac, PyTorch runs on CPU; Intel Iris is not a CUDA device. On Linux with NVIDIA, install the matching CUDA build using the [official selector](https://pytorch.org/get-started/locally/) and keep Torch/Torchvision versions paired. `device: auto` selects CUDA when available and CPU otherwise. CPU training on UA-DETRAC will be slow; reduce `batch_size`/`epochs` for a smoke run, or use a CUDA machine for full experiments.

## Run on Kaggle without storing the dataset locally

You do not need to download UA-DETRAC to your personal computer. Create or use a Kaggle Dataset containing the extracted image and annotation folders, then open a Kaggle Notebook and choose **Add Input** to attach that dataset (search by its Kaggle link/name). Kaggle mounts inputs read-only under `/kaggle/input`; this makes the files available to the notebook without copying them into your computer or downloading them on every run. The dataset must contain these folders, with the same nested layout as the local copy:

```text
DETRAC-Images/DETRAC-Images/
DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML/
DETRAC-Test-Annotations-XML/DETRAC-Test-Annotations-XML/
```

The configured source is [`bratjay/ua-detrac-orig`](https://www.kaggle.com/datasets/bratjay/ua-detrac-orig), recorded in `data/dataset_path.txt`. In the Kaggle Notebook, choose **Add Input** and attach that dataset. The loader resolves the identifier to `/kaggle/input/ua-detrac-orig`; if Kaggle shows a different folder name, set `TRAFFICVISION_DATA_ROOT` to that exact mounted path. It searches for the expected image and annotation folders at the mount root and up to two enclosing directory levels. It does not extract ZIP archives or copy the read-only input.

```python
!git clone https://github.com/sunnycloudhust/TrafficVision.git
%cd TrafficVision
!cat data/dataset_path.txt
!ls /kaggle/input/ua-detrac-orig
```

In a following notebook cell, run `!python scripts/train_pretrained.py` or `!python scripts/train_custom.py`. Kaggle's standard notebook image already includes PyTorch; avoid reinstalling the macOS-pinned Torch pair from `requirements.txt` there. Set `device: auto` in `configs/train.yaml` to use an available NVIDIA GPU. Kaggle input files are read-only, while checkpoints and logs go under the repository's `checkpoints/` and `outputs/` directories in `/kaggle/working`; save/download those outputs before the notebook session ends. If your Kaggle dataset contains ZIP archives rather than extracted folders, extract them to writable `/kaggle/working` storage first and put that extracted directory in `data/dataset_path.txt`. `TRAFFICVISION_DATA_ROOT` can also override the pointer file when needed.

ByteTrack uses Supervision's model-agnostic `Detections` adapter. The dependency is restricted below 0.31 because Supervision removed its built-in `ByteTrack` wrapper in 0.31. TrackEval is optional for tracking metrics:

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git external/TrackEval
python -m pip install -r external/TrackEval/requirements.txt
```

Set `trackeval_root: external/TrackEval` in `configs/evaluation.yaml` or pass `--trackeval-root` to the tracking script.

## Train

The scratch model sets `weights=None` and `weights_backbone=None`, so the backbone, FPN, proposal network, and RoI layers start randomly. The pretrained model starts from Torchvision's COCO Faster R-CNN ResNet-50-FPN weights; the backbone, FPN, RPN, and RoI box feature layers retain those weights, while the class-specific predictor is newly initialized. Optional backbone freezing is in `configs/model.yaml`.

```bash
python scripts/train_custom.py
python scripts/train_pretrained.py
```

Both commands use the same sequence split, batch size, epoch count, optimizer, and seed from YAML. Edit `configs/train.yaml` for learning rate, optimizer, weight decay, workers, checkpoints, resume path, or mixed precision. Mixed precision is CUDA-only. Training never changes batch size automatically after an out-of-memory error. The best checkpoint is selected by validation mAP@50; test data is not used for model selection.

For multiple GPUs, launch with PyTorch DistributedDataParallel (one process per GPU):

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_custom.py
torchrun --standalone --nproc_per_node=2 scripts/train_pretrained.py
```

DDP also supports CPU/Gloo when launched with multiple processes, though that is usually slower. Checkpoints and logs are written once by rank 0. Inference and evaluation load the unwrapped model and can run on one selected device.

Outputs: `checkpoints/{scratch,pretrained}/{best,last,epoch_*.pt}`, `outputs/logs/*_training.csv`, and loss curves. Checkpoints include model configuration, class mapping, seed, optimizer/scaler state, epoch, and Python/Torch/Torchvision versions.

## Detection evaluation

Evaluate the selected model on the official test split:

```bash
python scripts/evaluate_detection.py --checkpoint checkpoints/scratch/best.pt
python scripts/evaluate_detection.py --checkpoint checkpoints/pretrained/best.pt
```

The default IoU match threshold is 0.50. Predictions are sorted by confidence; each ground-truth object can match one prediction per image and class. Precision and recall use the configured confidence threshold. AP is calculated from the global ranked prediction list with a precision envelope, then mAP@50 is the unweighted mean over configured classes with test objects. The same test records and policy are used for both checkpoints. Raw predictions and JSON metrics are saved under `outputs/metrics/`.

The COCO-pretrained-only benchmark is optional because its label space differs. Supply a JSON map from Torchvision COCO category names to configured dataset classes; only mapped classes are scored:

```json
{"car": "car", "bus": "bus"}
```

```bash
python scripts/evaluate_coco_pretrained.py --mapping configs/coco_to_detrac.json
```

Do not map COCO `truck` to UA-DETRAC `van` or `others` unless that decision is valid for the experiment. The script records the explicit mapping with its results.

## Tracking and video

Run ByteTrack on a video; it creates one tracker per video and updates it once per frame, including frames with no detections:

```bash
python scripts/run_video.py input.mp4 --checkpoint checkpoints/pretrained/best.pt
```

Annotated MP4, frame/track CSV, and timing JSON are written under `outputs/videos/`. The adapter passes Faster R-CNN pixel boxes, confidence values, and contiguous class IDs into ByteTrack. `configs/tracking.yaml` controls activation threshold, lost-track buffer, matching threshold, and frame rate. IDs are reset for every video.

Evaluate tracking on the ground-truth sequence IDs:

```bash
python scripts/evaluate_tracking.py \
  --checkpoint checkpoints/pretrained/best.pt \
  --config configs/evaluation.yaml \
  --tracking-config configs/tracking.yaml \
  --trackeval-root external/TrackEval
```

The script exports MOT-format GT/predictions and invokes TrackEval for HOTA, IDF1, and MOTA. It treats the four vehicle classes as one foreground class for identity association, keeps the XML target IDs, and removes predicted boxes when at least half of a predicted box lies inside a UA-DETRAC ignored-region box. This is a documented class-agnostic TrackEval conversion; it is not a claim that the results reproduce the official UA-DETRAC benchmark protocol. Per-sequence and aggregate TrackEval output is retained with the MOT files.

## Compare results

After evaluating both trained checkpoints, make the requested CSV and Markdown table:

```bash
python scripts/compare_results.py \
  --scratch outputs/metrics/scratch_detection.json \
  --fine-tuned outputs/metrics/pretrained_detection.json
```

Pass `--pretrained-only outputs/metrics/coco_pretrained_only_detection.json` to include the mapped COCO-only run. Tracking metric JSON can be passed with `--scratch-tracking` and `--fine-tuned-tracking` when available. Missing metrics are emitted as `N/A`, never fabricated. Files are written to `outputs/metrics/comparison.csv` and `comparison.md`.

## Project layout and checks

`src/datasets` parses UA-DETRAC XML and COCO JSON; `src/models` builds scratch/pretrained Faster R-CNN; `src/training` owns training, resume, and checkpoints; `src/evaluation` computes detection metrics; `src/tracking` adapts detections to ByteTrack; `src/detection` and `src/visualization` run/render video. YAML files under `configs/` define data paths, classes, hyperparameters, thresholds, and output locations.

Run the unit suite after installing dependencies:

```bash
python -m pytest
```

Tests cover XML boxes and empty annotations, CPU model output/checkpoint round-trip, and the Faster R-CNN-to-ByteTrack adapter. Training and full-dataset metrics still require the installed model dependencies and are intentionally not run just by importing the project.
