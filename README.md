# TrafficVision — Deep Learning course project

Pipeline modular: **UA-DETRAC → detector → tracker → polygon zones → homography → speed → MP4/CSV/JSON**.

Code đã có đủ luồng train/evaluate/demo và kiểm thử CPU. Chưa có thí nghiệm train đầy đủ trên UA-DETRAC, checkpoint học tốt, camera calibration thực hoặc kết quả benchmark. Xem [báo cáo theo phase](docs/PROGRESS.md) trước khi dùng số liệu trong báo cáo môn học.

Yêu cầu gốc được giữ ở [AGENTS.md](AGENTS.md); file Markdown ban đầu cũng được giữ nguyên. Các lựa chọn kiến trúc và giới hạn ở [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. Cài đặt và chạy ngay

Python **3.10–3.12**. Máy hiện tại đã có `.venv` với bộ thư viện CPU đã kiểm tra.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU; với GPU hãy chọn torch/torchvision tương thích CUDA trước.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[train,tracking-eval,dev]"
# Ultralytics có thể cài opencv-python; cài contrib cuối để Selective Search có ximgproc.
python -m pip install --force-reinstall --no-deps opencv-contrib-python-headless==4.10.0.84

python scripts/run_trafficvision.py --help
python -m pytest -q
```

`requirements.txt` cài project cùng các dependency cho train, tracking evaluation và phát triển. Với GPU, hãy cài PyTorch/Torchvision tương thích CUDA trước. Thiết lập trên Colab xem [docs/COLAB.md](docs/COLAB.md).

## 2. Phase 1 — Chuẩn bị UA-DETRAC

Tải ảnh và XML từ [trang tác giả](https://sites.google.com/view/daweidu/projects/ua-detrac) vào ổ runtime, ví dụ `/content/ua_detrac_raw`; không cần Google Drive cá nhân. Google Drive share URL cần `gdown`; downloader HTTP trực tiếp không xử lý trang xác nhận của Google Drive.

```bash
python scripts/prepare_ua_detrac.py \
  --images /content/ua_detrac_raw/images \
  --annotations /content/ua_detrac_raw/annotations \
  --output data/ua_detrac --seed 42 --shard-size 500
```

Script dò thư mục ảnh chứa tên sequence (`MVI_...`), nhận `img00001.jpg` hoặc `00001.jpg`, parse XML và kiểm tra thiếu/ảnh trùng. Đầu ra:

| File | Vai trò |
|---|---|
| `frames.jsonl` | Manifest local: ảnh, boxes, labels, track IDs, ignored regions |
| `splits.json` | Danh sách video train/val/test; mặc định 70/15/15 theo seed |
| `train-*.tar`, `val-*.tar`, `test-*.tar` | Shard gồm cặp ảnh JPEG và annotation JSON liên tiếp |
| `shards.json` | URL, SHA256, dung lượng, split và số sample của từng shard |

Để giữ official test riêng, tự tạo `splits.json` rồi truyền `--splits path/to/splits.json`. Mọi XML đầu vào phải được bao phủ đúng một lần. Mặc định cần đủ video để cả ba split không rỗng. Không chia frame cùng video vào các split khác nhau.

Class mapping chung: `background=0`, **`car=1, bus=2, van=3, others=4`**. YOLO export đổi sang 0–3. Loại xe lạ gây lỗi để nhóm quyết định mapping; không tự đoán `truck` trong annotation thành class khác. Frame index XML giữ one-based; bbox từ continuous `left,top,width,height` đổi sang `xyxy`, không tự trừ 1 pixel. Cần overlay một số frame thật để xác nhận convention của nguồn tải.

### Remote-first

Upload là thao tác tùy chọn do nhóm chủ động chạy sau khi kiểm tra quyền phân phối dataset:

```bash
# Cấu hình HF_TOKEN bằng secret/environment, không ghi token vào source.
python scripts/upload_dataset_shards.py \
  --manifest data/ua_detrac/shards.json --repo-id YOUR_ORG/ua-detrac-shards
```

Repository HF mặc định private. Lưu file nhỏ `shards.remote.json` rồi sửa `configs/data/ua_detrac.yaml`:

```yaml
data:
  source: huggingface  # hoặc cloud cho HTTP/S3/GCS
  manifest: data/ua_detrac/shards.remote.json
  mode: auto
  cache_dir: null
  seed: 42
```

`stream` đọc TAR tuần tự, không giải nén toàn bộ; checksum ảnh kiểm tra lúc đọc. `cache` tải shard bằng retry, kiểm tra size/SHA256 và đổi tên atomically. `auto` theo `preferred_mode` trong manifest, mặc định `cache` để train nhiều epoch. `cache_dir: null` dùng thư mục tạm tự dọn khi source kết thúc; đặt `/content/trafficvision-cache` để dùng chung giữa lần chạy trong cùng runtime. HTTP và multiworker streaming đã được kiểm tra bằng server local; HF/S3 thực cần token/bucket của nhóm. S3/GCS cần extra `[cloud]` và biến môi trường backend tương ứng.

## 3. Phase 2 — Train và evaluate detector

Sửa `data.manifest`, `training.device` (`cpu`/`cuda`), batch size và đường dẫn output trong YAML. Tất cả đường dẫn dữ liệu/output tính từ thư mục repo; `extends` tính tương đối với file YAML đang chứa nó.

```bash
python scripts/run_trafficvision.py train --model faster_rcnn --init random \
  --config configs/experiments/faster_rcnn_scratch.yaml

python scripts/run_trafficvision.py train --model fast_rcnn --init random \
  --config configs/experiments/fast_rcnn_scratch.yaml

python scripts/run_trafficvision.py train --model rcnn --init random \
  --config configs/experiments/rcnn_scratch.yaml

python scripts/run_trafficvision.py train --model rtdetr --init random \
  --config configs/experiments/rtdetr_scratch.yaml

python scripts/run_trafficvision.py train --model yolov8 --init pretrained \
  --config configs/experiments/yolov8_baseline.yaml
```

Custom detectors tự lắp kiến trúc; mặc định CNN nhỏ train từ random để dễ debug. Faster/Fast/R-CNN hỗ trợ `params.backbone: resnet18`, `--init imagenet` cho transfer learning; không gọi đó là scratch. RT-DETR là **biến thể học tập rút gọn**, xem kiến trúc trước khi trình bày kết quả. R-CNN/Fast R-CNN mặc định dùng Selective Search; `proposal_strategy: grid` chỉ dành cho kiểm thử tự động nhẹ.

Custom runs lưu `config.yaml`, `metrics.jsonl`, `last.pt`, `best.pt` (chọn theo validation COCO mAP). Checkpoint gồm model params, state, optimizer, scheduler, epoch, seed/config và RNG; resume tại ranh giới epoch:

```bash
python scripts/run_trafficvision.py train --config configs/experiments/faster_rcnn_scratch.yaml \
  --resume outputs/faster_rcnn/last.pt

python scripts/run_trafficvision.py evaluate --model faster_rcnn \
  --checkpoint outputs/faster_rcnn/best.pt --split test \
  --config configs/evaluation/ua_detrac.yaml --output outputs/evaluation/faster_rcnn.json
```

YOLO materialize split vào ổ runtime để dùng trainer native. Weights sau fine-tune ở `outputs/yolov8/train/weights/best.pt`. Dùng cùng lệnh `evaluate`, thay `--model yolov8` và checkpoint tương ứng để so sánh bằng evaluator chung. COCO pretrained không có class `van`; adapter kiểm tra tên class checkpoint, map `car→car`, `bus→bus`, `truck→others` và ghi hạn chế này trong metrics.

Evaluation báo mAP@.5, mAP@.5:.95, AP theo class, precision/recall tại threshold .25, thời gian predict, parameter count/bytes và CUDA peak memory nếu có. Cố định split, ignore policy, threshold và phần cứng khi so sánh. Không dùng test để tuning.

## 4. Phase 3 — Tracking và polygon

Tracker config ở `configs/trackers/`: `sort`, `bytetrack`, `botsort`. Đổi tracker không cần sửa model. SORT là biến thể Kalman xyxy+velocity; ByteTrack/BoT-SORT dùng adapter Ultralytics pin version. BoT-SORT mặc định camera cố định, GMC off, ReID off; bản adapter hiện tại từ chối bật ReID khi chưa có encoder.

Để đo tracking trên **toàn split**, không cần ghép ảnh thành MP4:

```bash
python scripts/run_trafficvision.py track-dataset --model faster_rcnn \
  --checkpoint outputs/faster_rcnn/best.pt --tracker bytetrack --split test \
  --config configs/evaluation/ua_detrac.yaml --output outputs/evaluation/faster_bytetrack.csv
```

Lệnh reset tracker giữa các video, kiểm tra thứ tự frame, xuất trajectories và chạy TrackEval **HOTA, IDF1, MOTA, ID switches, fragments** bằng GT IDs thực. Kết quả theo protocol class-aware của project; không phải leaderboard PR-MOTA chính thức UA-DETRAC. Các metric tỷ lệ lưu theo khoảng 0–1, MOTA có thể âm. Cần chạy đủ video để so sánh.

Polygon mẫu `configs/zones/MVI_20011.yaml` chưa phải calibration thật. Vẽ lại cho video rồi chỉnh `frame_size`, `confidence_threshold`, `inference_scale`, `measure_speed`. Dùng bottom-center để gán vùng; crop lượt riêng cho vùng có scale >1 rồi đưa box về frame gốc và NMS theo class. Threshold vùng thấp giữ candidate cho association; `new_track_thresh` vẫn kiểm soát việc sinh ID mới.

Có thể chạy zone labeler không cần tham số. Chương trình tự lấy frame đầu tiên
trong `data/ua_detrac` làm frame mẫu và lưu polygon dùng chung vào
`configs/zones/common.yaml`:

```bash
python data-labelling/zone_labeler.py
```

Zone dùng chung không gắn với một `camera_id` cụ thể nên có thể dùng cho các video
cùng góc quay.

## 5. Phase 4 — Calibration và vận tốc

Với mỗi sequence, lưu tọa độ thực đã đo bằng **mét** tại `data/ua_detrac/world_points/world_points_<VIDEO_ID>.json`. Các điểm ảnh và điểm thực phải có cùng thứ tự, gồm ít nhất 4 điểm không thẳng hàng trên mặt đường. Không tự suy ra khoảng cách thực từ bbox.

```bash
python scripts/calibrate_camera.py \
  --image data/ua_detrac/raw/DETRAC/DETRAC-Images/DETRAC-Images/MVI_20011/img00001.jpg \
  --interactive --world-points data/ua_detrac/world_points/world_points_MVI_20011.json \
  --camera-id MVI_20011 --fps 25 --output configs/cameras/MVI_20011.yaml
```

Desktop có OpenCV GUI có thể thay `--image-points` bằng `--interactive`. Colab/headless dùng JSON. Không cài đồng thời nhiều biến thể OpenCV khi muốn dùng GUI; xem hướng dẫn môi trường. Kiểm tra phép chiếu trên điểm độc lập: reprojection thấp ở đúng bốn điểm fit không chứng minh calibration chính xác.

Pipeline dùng Kalman constant-velocity trên tọa độ mét; Jacobian homography truyền nhiễu pixel sang world coordinates. Speed trả `N/A` khi chưa đủ quan sát, uncertainty cao, timestamp lỗi, gap dài, innovation bị loại hoặc chưa có homography. Các tham số nằm trong `speed` của camera YAML; module hồi quy cũ vẫn được giữ để đối chiếu. Chỉ dùng quan sát detection hiện tại; không lấy vị trí track dự đoán qua occlusion làm phép đo.

```bash
python scripts/run_trafficvision.py evaluate-speed --pairs /path/to/matched_speed_pairs.csv
```

CSV cần cặp đã đối chiếu cùng phương tiện/thời gian: `gt_kmh,predicted_kmh`. Chỉ báo cáo MAE/RMSE/MAPE thực khi có GT speed; MAPE bỏ các GT bằng 0.

## 6. Phase 5 — Video demo

```bash
python scripts/run_trafficvision.py demo --source /path/to/traffic.mp4 \
  --model faster_rcnn --checkpoint outputs/faster_rcnn/best.pt --tracker bytetrack \
  --camera-config configs/cameras/MVI_20011.yaml --zone-config configs/zones/MVI_20011.yaml \
  --output outputs/demo/faster_bytetrack.mp4
```

Có thể bỏ camera/zone config để xem detection + tracking, speed `N/A`. Video pretrained chưa fine-tune:

```bash
python scripts/run_trafficvision.py demo --source /path/to/traffic.mp4 --model yolov8 \
  --config configs/models/yolov8.yaml --tracker bytetrack --output outputs/demo/yolo_coco.mp4
```

MP4 có bbox/class/ID/speed/zones; CSV và JSON lưu `video_id, frame_id, timestamp, track_id, label, class, confidence, bbox, world_x/y, speed_kmh, zones`. File `.summary.json` ghi FPS, frame size và trạng thái calibration. `--timestamp-mode pts` ưu tiên presentation timestamps cho video có timestamp đáng tin cậy và kiểm tra đơn điệu; mặc định `fps` phù hợp UA-DETRAC constant-FPS. `--video-id MVI_...` dùng khi tên file video khác sequence GT.

Tổng hợp số đo đã có:

```bash
python scripts/compare_runs.py outputs/evaluation/faster_rcnn.json \
  outputs/evaluation/faster_bytetrack.metrics.json --output outputs/comparison.csv
```

## Tổ chức code và phân công

| Thư mục | Công việc |
|---|---|
| `data/` trong package | Parser, sources, shard/cache, split, PyTorch dataset |
| `models/`, `training/` | Kiến trúc, baseline, loss, train/resume |
| `tracking/`, `zones/` | Association và phân vùng camera |
| `geometry/`, `speed/` | Calibration và vận tốc theo timestamp |
| `evaluation/` | Metric detection/tracking/speed |
| `pipeline/`, `visualization/` | Điều phối, video và export |
| `configs/`, `scripts/`, `tests/` | Thí nghiệm, công cụ và regression tests |

Nhóm có thể chia theo các hàng trên; mỗi model/tracker phải giữ interface chung. Không commit dataset, checkpoint, token, output hoặc `.venv`. Làm việc trên branch riêng và gửi pull request để nhóm review trước khi merge.
