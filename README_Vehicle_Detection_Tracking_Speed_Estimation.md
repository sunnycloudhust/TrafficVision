# Vehicle Detection, Multi-Object Tracking & Speed Estimation

> **Deep Learning Course Project**  
> Phát hiện phương tiện, theo dõi ID và ước lượng vận tốc từ video giao thông bằng Computer Vision và Deep Learning.

## 1. Giới thiệu

Project xây dựng một pipeline xử lý video giao thông gồm bốn bước chính:

1. **Vehicle Detection:** phát hiện xe trong từng frame.
2. **Multi-Object Tracking (MOT):** gán và duy trì ID cho từng phương tiện qua nhiều frame.
3. **Polygon Zones:** xác định vùng quan tâm trên camera, phân vùng gần/xa và điều chỉnh xử lý theo từng vùng.
4. **Speed Estimation:** ánh xạ vị trí xe từ hệ tọa độ ảnh sang mặt phẳng đường bằng Perspective Transformation (Homography), sau đó tính vận tốc theo thời gian.

Dataset chính là **UA-DETRAC**. Các mô hình tự xây dựng sẽ được huấn luyện trên dataset này; những mô hình pretrained như YOLOv8 được dùng làm baseline để so sánh.

### Mục tiêu đầu ra

Sau khi hoàn thành, project cần hỗ trợ:

- Huấn luyện các detector trên UA-DETRAC.
- Đánh giá detector trên tập validation/test.
- Chạy detection + tracking với nhiều tracker có thể thay thế độc lập.
- Hiệu chuẩn camera và ước lượng vận tốc phương tiện.
- Chạy một video demo có sẵn hoặc video được cung cấp qua đường dẫn.
- Xuất video MP4 có bounding box, class, tracking ID, vận tốc (km/h) và polygon zones.
- Xuất kết quả dạng CSV/JSON để phân tích từng phương tiện và quỹ đạo.

> **Phạm vi quan trọng:** UA-DETRAC là dataset detection/tracking. Để báo cáo sai số vận tốc thực (km/h), cần có thông tin hiệu chuẩn mặt phẳng đường và vận tốc ground-truth tương ứng. Không nên coi bounding box hoặc tracking ID đơn thuần là ground-truth vận tốc.

---

## 2. Dataset: UA-DETRAC

### 2.1. Thông tin

UA-DETRAC là benchmark cho vehicle detection và multi-object tracking, gồm khoảng 100 video, hơn 140.000 frame, độ phân giải ảnh 960×540 và tốc độ ghi hình 25 FPS. Dataset có bounding box và thông tin quỹ đạo/ID phương tiện cùng một số thuộc tính như loại xe, điều kiện ánh sáng, occlusion và truncation.

Các nhóm phương tiện thường được sử dụng gồm `car`, `bus`, `van` và `others`. Cần kiểm tra annotation gốc khi viết bộ chuyển đổi nhãn để cố định class mapping cho toàn bộ thí nghiệm.

### 2.2. Nguồn tải dữ liệu

- **Trang dataset của tác giả:** <https://sites.google.com/view/daweidu/projects/ua-detrac>
- **DETRAC-Images trên Google Drive:** <https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view?usp=sharing>
- **DETRAC-Annos trên Google Drive:** <https://drive.google.com/drive/folders/1am_x3ywLJnABq4REtzbi0XTu5csFrp8v?usp=sharing>
- Trang chính thức của tác giả cũng cung cấp các mục download annotation, detection và toolkit. Hãy ưu tiên kiểm tra link từ trang này nếu một mirror không còn hoạt động.

**Lưu ý:** dataset rất lớn. Không commit dataset, file ZIP, frame ảnh hoặc checkpoint vào Git repository.

### 2.3. Chiến lược truy cập dataset: remote-first

Mục tiêu là có thể clone code trên máy mới/Colab, cấu hình quyền truy cập và bắt đầu train mà không cần chuẩn bị dataset thủ công trên máy cá nhân. Tuy nhiên, GPU vẫn phải đọc dữ liệu trong lúc train. Vì vậy, “không lưu dataset trên máy cá nhân/Google Drive” được triển khai theo nghĩa **không cần lưu bản dataset lâu dài**; runtime có thể dùng bộ nhớ đệm tạm thời để tăng tốc.

Project nên hỗ trợ ba chế độ:

| Chế độ | Cách hoạt động | Khi nên dùng |
|---|---|---|
| `stream` | Đọc dữ liệu theo shard từ Hugging Face Dataset hoặc object storage (S3-compatible/HTTP) bằng streaming/DataPipe. | Khi nguồn dữ liệu hỗ trợ truy cập file/shard từ xa và có băng thông ổn định. |
| `cache` | Tải các shard cần dùng vào thư mục tạm của runtime, ví dụ `/content` hoặc `/tmp`, rồi train từ cache. Cache có thể bị xóa khi runtime kết thúc. | Khuyến nghị cho train nhiều epoch vì hạn chế tải lặp lại. |
| `auto` | Ưu tiên remote streaming; nếu cấu hình nguồn yêu cầu cache thì tự chuẩn bị cache tạm. | Chế độ mặc định khi chạy trên Colab/máy khác. |

#### Phương án khuyến nghị cho project

1. Tải bộ dữ liệu gốc **một lần** từ nguồn chính thức của UA-DETRAC.
2. Viết preprocessing để chuyển ảnh và annotation sang định dạng chuẩn của project.
3. Chia dữ liệu thành các shard có kích thước vừa phải (ví dụ WebDataset `.tar` hoặc Parquet kèm đường dẫn ảnh).
4. Đưa các shard đã xử lý lên Hugging Face Dataset hoặc cloud object storage.
5. Lưu `repo_id`/địa chỉ bucket và metadata trong file config; không hard-code token hay mật khẩu.
6. Khi chạy trên Colab/máy khác, chỉ cần cấu hình token/quyền đọc. DataLoader đọc shard từ xa và có thể cache tạm tại runtime.

Cách này giúp tránh việc mỗi lần chạy phải tự tải và giải nén toàn bộ bộ dữ liệu gốc. Nếu dùng Hugging Face, cần đóng gói dữ liệu theo định dạng hỗ trợ streaming; chỉ gọi `snapshot_download()` hoặc `hf_hub_download()` với archive lớn vẫn có thể tải toàn bộ file về runtime, không phải streaming thực sự.

#### Ví dụ tải dữ liệu gốc vào ổ đĩa tạm của Colab

Đây là phương án fallback đơn giản. Dữ liệu được đặt trên ổ đĩa runtime `/content`, không phải Google Drive cá nhân. Dung lượng runtime vẫn phải đủ cho phần dữ liệu cần tải và giải nén.

```bash
# Cài công cụ tải từ Google Drive
pip install -q gdown

# Tạo thư mục tạm trên Colab runtime
mkdir -p /content/ua_detrac_raw

# Tải archive ảnh từ link chính thức của tác giả
gdown --fuzzy \
  "https://drive.google.com/file/d/1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm/view?usp=sharing" \
  -O /content/ua_detrac_raw/DETRAC-Images.zip

# Giải nén vào runtime (chỉ thực hiện khi cần chế độ cache)
unzip -q /content/ua_detrac_raw/DETRAC-Images.zip \
  -d /content/ua_detrac_raw/images
```

Annotation được tải từ thư mục `DETRAC-Annos` ở link phía trên. Cần kiểm tra tên file và cấu trúc thư mục thực tế trước khi viết parser; không giả định mọi mirror có cùng cấu trúc.

> Google Drive có thể giới hạn quota, yêu cầu quyền truy cập hoặc thay đổi cơ chế tải. Vì thế downloader cần có kiểm tra file tồn tại, dung lượng, checksum (nếu có), retry và thông báo lỗi rõ ràng. Không nên ghi token riêng tư vào README hoặc source code.

### 2.4. Dataset adapter và annotation parser

Tạo một interface chung để các nguồn dữ liệu không phụ thuộc vào model:

```text
DatasetSource
  ├── LocalDatasetSource
  ├── HuggingFaceStreamingSource
  └── CloudObjectStorageSource

UADETRACDataset
  ├── đọc frame/ảnh
  ├── đọc bounding boxes và class labels
  ├── đọc track IDs / trajectory annotations
  └── trả dữ liệu theo format thống nhất
```

Mỗi sample detection nên có tối thiểu:

```python
{
    "image": image,
    "boxes": FloatTensor[N, 4],  # x1, y1, x2, y2
    "labels": Int64Tensor[N],
    "image_id": image_id,
}
```

Thông tin tracking cần được giữ riêng hoặc bổ sung `track_ids` tương ứng với từng bounding box. Parser cần xử lý đúng frame index, video ID, coordinate convention và các vùng annotation bị ignore nếu có.

### 2.5. Split dữ liệu

- Tạo train/validation/test theo **video sequence**, không chia ngẫu nhiên frame của cùng một video sang nhiều split.
- Lưu danh sách video cho mỗi split trong config/manifest để mọi lần chạy có thể tái lập.
- Không dùng test set để chọn hyperparameter hoặc điều chỉnh threshold.
- Khi báo cáo kết quả, ghi rõ split, số video, số frame và cách chuyển đổi annotation.

---

## 3. Object Detection

Project có hai nhóm detector riêng biệt: **pretrained baseline** và **custom detector tự xây dựng**.

### 3.1. Pretrained baseline

Dùng các model pretrained có sẵn để tạo baseline, tối thiểu:

- **YOLOv8**: baseline một-stage, tốc độ suy luận tốt; có thể fine-tune trên UA-DETRAC.
- Có thể mở rộng thêm YOLO phiên bản khác hoặc RT-DETR pretrained để so sánh.

Baseline cần ghi rõ:

- Tên model và kích thước (`n`, `s`, `m`, ...).
- Checkpoint/pretrained weights và nguồn tải.
- Cách fine-tune, số epoch, image size, batch size và optimizer.
- Kết quả trên cùng validation/test split với các detector còn lại.

Ví dụ lệnh mục tiêu (cần được hiện thực trong CLI của project):

```bash
python -m trafficvision.cli train \
  --model yolov8 \
  --config configs/experiments/yolov8_baseline.yaml
```

### 3.2. Các detector tự xây dựng

Các model sau phải có module kiến trúc riêng trong codebase, có thể khởi tạo **random initialization** để train từ đầu trên UA-DETRAC. Không được chỉ gọi một model pretrained có sẵn rồi đổi tên thành custom model.

| Model | Thành phần chính | Vai trò |
|---|---|---|
| R-CNN | Region proposals (ví dụ Selective Search), CNN feature extraction, classifier và bounding-box regressor. | Mốc so sánh/triển khai nguyên lý two-stage detector cổ điển; dự kiến chậm. |
| Fast R-CNN | CNN backbone dùng chung cho ảnh, region proposals, RoI Pooling/RoI Align, classification head và box-regression head. | So sánh cải tiến so với R-CNN. |
| Faster R-CNN | Backbone, Region Proposal Network (RPN), RoI feature extractor, classification head và box-regression head. | Detector two-stage chính của project. |
| RT-DETR | CNN backbone, hybrid encoder, Transformer decoder, query selection và set-based loss/Hungarian matching. | Detector Transformer để so sánh với nhóm two-stage và YOLO. |

#### Quy ước khởi tạo

- `random`: khởi tạo toàn bộ trọng số model từ đầu; dùng để đáp ứng thí nghiệm train-from-scratch.
- `imagenet`: có thể nạp pretrained weights cho backbone, trong khi detection head được khởi tạo mới. Đây là transfer learning, không phải train hoàn toàn từ đầu.
- Baseline YOLOv8 pretrained được quản lý riêng, không gộp vào custom model registry.

#### Thứ tự triển khai đề xuất

1. Faster R-CNN (mô hình trọng tâm, phù hợp phân công hiện tại).
2. YOLOv8 pretrained baseline.
3. Fast R-CNN và R-CNN để so sánh kiến trúc/chi phí tính toán.
4. RT-DETR custom implementation.

R-CNN và Fast R-CNN có thể rất tốn thời gian do region proposal/feature extraction. Nên triển khai chúng như các thí nghiệm có cấu hình riêng, không để chúng chặn pipeline chính.

### 3.3. Chuẩn hóa class và output

Tất cả detector phải trả về cùng một format để tracker không phụ thuộc model:

```python
{
    "boxes": Tensor[N, 4],  # xyxy trên ảnh gốc
    "scores": Tensor[N],
    "labels": Tensor[N],
}
```

Cần thống nhất class mapping, quy ước tọa độ, confidence score, NMS và cách scale bounding box về kích thước frame gốc.

---

## 4. Polygon Zones và xử lý phương tiện ở xa

Sử dụng `PolygonZone` (có thể triển khai bằng `supervision.PolygonZone`) để khai báo các vùng đa giác trên frame camera.

### 4.1. Mục đích

- Chỉ xử lý/đếm các phương tiện nằm trong vùng đường quan tâm.
- Phân chia khu vực theo phối cảnh: `far_zone`, `mid_zone`, `near_zone`.
- Thiết lập confidence threshold, input resolution hoặc inference strategy khác nhau theo từng vùng.
- Hỗ trợ vùng đo tốc độ và vùng kích hoạt sự kiện riêng.

Có thể dùng điểm neo `BOTTOM_CENTER` của bounding box làm vị trí tiếp xúc gần mặt đường để kiểm tra một detection thuộc polygon nào.

### 4.2. Xử lý xe nhỏ/xa camera

Xe ở xa thường chỉ chiếm ít pixel nên detector có thể bỏ sót hoặc cho confidence thấp. PolygonZone **không tự tạo detection và không thể tự khôi phục một track đã mất**. Vì vậy cần phối hợp nhiều kỹ thuật:

- Chia polygon thành vùng xa/gần và dùng threshold phù hợp cho vùng xa.
- Tăng input resolution hoặc dùng tiled inference/crop cho khu vực xa.
- Điều chỉnh NMS để hạn chế loại nhầm các bounding box nhỏ/đứng gần nhau.
- Dùng confidence thấp có kiểm soát cho bước association của ByteTrack.
- Giữ track trong một khoảng thời gian ngắn khi bị mất detection; chỉ re-activate khi có detection phù hợp về vị trí/chuyển động.
- Đánh giá riêng recall theo kích thước đối tượng hoặc theo vùng xa/gần.

Không nên giảm threshold toàn ảnh một cách tùy tiện vì có thể làm tăng false positives.

### 4.3. Cấu hình polygon

Polygon phải được định nghĩa theo đúng độ phân giải frame camera. Ví dụ cấu hình minh họa:

```yaml
camera_id: camera_01
frame_size: [960, 540]
zones:
  far_zone:
    polygon: [[320, 80], [640, 80], [760, 260], [220, 260]]
    confidence_threshold: 0.15
    inference_scale: 1.5
  near_zone:
    polygon: [[80, 260], [880, 260], [960, 540], [0, 540]]
    confidence_threshold: 0.30
```

Tọa độ ở trên chỉ là ví dụ minh họa, **không phải polygon đã hiệu chuẩn cho UA-DETRAC**. Mỗi video/camera cần polygon riêng và phải kiểm tra trực quan trước khi chạy.

---

## 5. Multi-Object Tracking

Tracker nhận detection chuẩn hóa từ bất kỳ detector nào. Thiết kế adapter chung để có thể đổi tracker mà không phải sửa code của model.

### Thuật toán cần hỗ trợ

- **SORT:** baseline đơn giản sử dụng Kalman Filter và IoU-based association.
- **ByteTrack:** association hai giai đoạn, tận dụng cả detection confidence thấp để khôi phục liên kết với track đang tồn tại.
- **BoT-SORT:** mở rộng tracking với camera-motion compensation và tùy chọn Re-Identification (ReID).

Có thể mở rộng OC-SORT/Deep OC-SORT nếu còn thời gian.

### Quy trình tracking

```text
Detector outputs
      │
      ▼
Detection filtering / class filtering
      │
      ▼
Tracker adapter (SORT / ByteTrack / BoT-SORT)
      │
      ▼
Track objects: track_id + bbox + class + confidence
      │
      ▼
Polygon-zone assignment
      │
      ▼
World-coordinate projection and speed estimation
```

Nên bắt đầu bằng ByteTrack cho camera cố định; sau đó so sánh BoT-SORT. Với BoT-SORT, có thể tắt global motion compensation nếu camera cố định để giảm overhead, và chỉ bật ReID khi cần thiết.

---

## 6. Speed Estimation bằng Perspective Transformation

### 6.1. Ý tưởng

Bounding box nằm trong hệ tọa độ pixel, không thể lấy trực tiếp khoảng cách pixel chia cho thời gian để có vận tốc km/h. Cần ánh xạ điểm xe trên mặt đường từ ảnh sang hệ tọa độ mặt phẳng thực.

Dùng **Homography / Perspective Transformation**:

\[
\lambda\begin{bmatrix}X\\Y\\1\end{bmatrix}
= H\begin{bmatrix}u\\v\\1\end{bmatrix}
\]

Trong đó:

- `(u, v)`: tọa độ điểm trên ảnh.
- `(X, Y)`: tọa độ tương ứng trên mặt phẳng đường (đơn vị mét).
- `H`: ma trận homography 3×3.
- `λ`: hệ số tỷ lệ đồng nhất.

Điểm đại diện cho phương tiện nên lấy tại **bottom-center** của bounding box thay vì tâm bbox, vì điểm này gần vị trí bánh xe tiếp xúc với mặt đường hơn.

Sau khi chuyển sang tọa độ mặt đường:

\[
d_t = \sqrt{(X_t-X_{t-1})^2+(Y_t-Y_{t-1})^2}
\]

\[
v_{m/s}=\frac{d_t}{\Delta t}, \qquad
v_{km/h}=3.6\times v_{m/s}
\]

Với video FPS cố định và frame cách nhau `k` frame, `Δt = k / FPS`. Nếu timestamp có sẵn thì ưu tiên sử dụng timestamp.

### 6.2. Hiệu chuẩn camera

Mỗi camera/video cần một file calibration riêng:

1. Chọn tối thiểu 4 điểm không thẳng hàng trên mặt phẳng đường trong ảnh.
2. Xác định tọa độ thực tương ứng của các điểm đó (mét), dựa trên khoảng cách đã biết hoặc phép đo tại hiện trường.
3. Tính homography bằng `cv2.findHomography()` hoặc `cv2.getPerspectiveTransform()`.
4. Kiểm tra bằng cách chiếu các điểm đã biết và đo sai lệch.
5. Lưu ma trận `H`, frame size, FPS và đơn vị trong file config camera.

Ví dụ cấu hình:

```yaml
camera_id: camera_01
fps: 25
frame_size: [960, 540]
world_unit: meter
homography:
  # Thay bằng ma trận đã hiệu chuẩn cho video thực tế.
  matrix: null
speed:
  smoothing_window: 5
  max_reasonable_kmh: 160
  min_track_observations: 3
```

Không được điền ma trận homography giả định để báo cáo vận tốc thực. Homography chỉ phù hợp khi các điểm được chiếu nằm trên cùng mặt phẳng (xấp xỉ mặt đường); các đoạn đường dốc/cao độ thay đổi mạnh có thể cần mô hình hiệu chuẩn khác.

### 6.3. Làm mượt và kiểm soát kết quả

- Tính vận tốc từ các quan sát của cùng `track_id`.
- Dùng moving average/EMA hoặc cửa sổ nhiều frame để giảm nhiễu bbox.
- Bỏ qua track quá ngắn hoặc có timestamp không hợp lệ.
- Kiểm tra vận tốc bất thường và các bước nhảy do ID switch.
- Khi track bị mất rồi được khôi phục, không tính vận tốc bằng khoảng thời gian/frame sai lệch.

### 6.4. Đánh giá vận tốc

Nếu có ground-truth speed, dùng MAE, RMSE và MAPE (chỉ tính MAPE khi ground-truth khác 0). Nếu chỉ có UA-DETRAC mà không có ground-truth tốc độ theo mét/giây, chỉ nên báo cáo demo định tính hoặc đánh giá độ nhất quán sau calibration; không tuyên bố đây là sai số vận tốc có kiểm chứng.

---

## 7. Training, Evaluation và Demo

### 7.1. Training

Các thí nghiệm cần lưu đầy đủ config, seed, dataset split, model initialization mode, optimizer, learning rate, scheduler, batch size, image size, epoch và checkpoint tốt nhất.

Ví dụ lệnh mục tiêu:

```bash
# Train Faster R-CNN từ random initialization
python -m trafficvision.cli train \
  --model faster_rcnn \
  --init random \
  --config configs/experiments/faster_rcnn_scratch.yaml

# Train custom RT-DETR từ random initialization
python -m trafficvision.cli train \
  --model rtdetr \
  --init random \
  --config configs/experiments/rtdetr_scratch.yaml

# Fine-tune YOLOv8 pretrained baseline
python -m trafficvision.cli train \
  --model yolov8 \
  --init pretrained \
  --config configs/experiments/yolov8_baseline.yaml
```

### 7.2. Evaluation

```bash
python -m trafficvision.cli evaluate \
  --model faster_rcnn \
  --checkpoint outputs/faster_rcnn/best.pt \
  --split test \
  --config configs/evaluation/ua_detrac.yaml
```

Metrics đề xuất:

| Thành phần | Metrics |
|---|---|
| Detection | Precision, Recall, mAP@0.5, mAP@0.5:0.95 |
| Tracking | HOTA, IDF1, MOTA, ID Switches, số track bị fragment |
| Speed | MAE, RMSE, MAPE (khi có ground-truth speed) |
| Runtime | FPS, inference latency, GPU memory và model size |

Khi so sánh model, sử dụng cùng test split, class mapping và điều kiện đánh giá. Với tracking, cần dùng ground-truth trajectory IDs tương ứng và evaluator phù hợp; không đánh giá tracking chỉ bằng số lượng ID nhìn thấy trên video.

### 7.3. Chạy video demo

Sau khi có checkpoint và calibration cho camera:

```bash
python -m trafficvision.cli demo \
  --source /path/to/traffic_video.mp4 \
  --model faster_rcnn \
  --checkpoint outputs/faster_rcnn/best.pt \
  --tracker bytetrack \
  --camera-config configs/cameras/camera_01.yaml \
  --zone-config configs/zones/camera_01.yaml \
  --output outputs/demo/fasterrcnn_bytetrack_speed.mp4
```

Có thể thay `--model faster_rcnn` bằng `yolov8`, `fast_rcnn`, `rcnn` hoặc `rtdetr`; thay `--tracker bytetrack` bằng `botsort` hoặc `sort`.

Video kết quả cần hiển thị:

- Bounding box và class phương tiện.
- Tracking ID ổn định theo thời gian.
- Vận tốc km/h (hoặc `N/A` khi chưa đủ dữ liệu/calibration).
- Polygon zones và nhãn vùng.
- FPS hoặc thông tin runtime (tùy chọn).

Đồng thời lưu CSV/JSON gồm `video_id`, `frame_id`, `timestamp`, `track_id`, `class`, bbox, tọa độ mặt đường và speed.

---

## 8. Kiến trúc codebase

Code cần chia thành module theo trách nhiệm. Không đặt toàn bộ dataset loader, model, training loop, tracking, speed estimation và visualization trong một file lớn.

```text
vehicle-speed-estimation/
│
├── README.md
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── train.txt
│   └── dev.txt
│
├── configs/
│   ├── data/
│   │   └── ua_detrac.yaml
│   ├── models/
│   │   ├── rcnn.yaml
│   │   ├── fast_rcnn.yaml
│   │   ├── faster_rcnn.yaml
│   │   ├── rtdetr.yaml
│   │   └── yolov8.yaml
│   ├── trackers/
│   │   ├── sort.yaml
│   │   ├── bytetrack.yaml
│   │   └── botsort.yaml
│   ├── zones/
│   │   └── camera_01.yaml
│   ├── cameras/
│   │   └── camera_01.yaml
│   └── experiments/
│       ├── faster_rcnn_scratch.yaml
│       └── yolov8_baseline.yaml
│
├── src/trafficvision/
│   ├── cli.py
│   ├── registry.py
│   │
│   ├── data/
│   │   ├── sources/
│   │   │   ├── base.py
│   │   │   ├── local.py
│   │   │   ├── huggingface_stream.py
│   │   │   └── cloud_storage.py
│   │   ├── datasets/
│   │   │   └── ua_detrac.py
│   │   ├── parsers/
│   │   │   ├── xml_parser.py
│   │   │   └── trajectory_parser.py
│   │   ├── cache.py
│   │   ├── transforms.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   ├── base.py
│   │   ├── backbones/
│   │   ├── detectors/
│   │   │   ├── rcnn/
│   │   │   ├── fast_rcnn/
│   │   │   ├── faster_rcnn/
│   │   │   └── rtdetr/
│   │   └── baselines/
│   │       └── yolov8.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── detection_metrics.py
│   │   ├── tracking_metrics.py
│   │   └── speed_metrics.py
│   │
│   ├── tracking/
│   │   ├── base.py
│   │   ├── sort.py
│   │   ├── bytetrack.py
│   │   └── botsort.py
│   │
│   ├── zones/
│   │   ├── polygon_zone.py
│   │   └── zone_manager.py
│   │
│   ├── geometry/
│   │   ├── calibration.py
│   │   ├── homography.py
│   │   └── coordinates.py
│   │
│   ├── speed/
│   │   ├── estimator.py
│   │   └── smoothing.py
│   │
│   ├── pipeline/
│   │   ├── detection_tracking.py
│   │   └── video_pipeline.py
│   │
│   └── visualization/
│       ├── annotator.py
│       ├── video_writer.py
│       └── result_exporter.py
│
├── scripts/
│   ├── prepare_ua_detrac.py
│   ├── upload_dataset_shards.py
│   └── calibrate_camera.py
│
├── tests/
│   ├── test_dataset_parser.py
│   ├── test_model_output.py
│   ├── test_polygon_zone.py
│   ├── test_homography.py
│   └── test_speed_estimator.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── data/                 # cache/local data; không commit
├── outputs/              # checkpoints, logs, videos; không commit
└── .gitignore
```

### Nguyên tắc thiết kế

- `data/` không được import trực tiếp model implementation.
- Detector chỉ phát hiện đối tượng, không tự chạy tracker hoặc speed estimator.
- Tracker chỉ nhận detections theo interface chung và trả tracking IDs.
- `geometry/` chịu trách nhiệm calibration và phép chiếu tọa độ.
- `speed/` chỉ tính/làm mượt vận tốc từ track history và world coordinates.
- `zones/` quản lý polygon và logic phân vùng; không trộn vào model backbone.
- `pipeline/` điều phối các module, không chứa implementation chi tiết của từng thuật toán.
- Các tham số thí nghiệm đặt trong YAML; tránh hard-code trong Python.
- Dùng registry/factory để chọn model, tracker và dataset bằng config.

---

## 9. Cài đặt môi trường

Nên sử dụng Python virtual environment hoặc Colab runtime. Phiên bản PyTorch/CUDA cần chọn phù hợp với GPU và runtime đang dùng.

```bash
# Clone project
git clone <YOUR_REPOSITORY_URL>
cd vehicle-speed-estimation

# Tạo môi trường ảo (local)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

# Cài dependencies (các extra cần được khai báo trong pyproject.toml)
pip install --upgrade pip
pip install -e ".[train]"
```

Các nhóm thư viện dự kiến:

- Core: `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `PyYAML`, `tqdm`.
- Baseline: `ultralytics`.
- Polygon/visualization: `supervision`.
- Dataset/cloud: `huggingface_hub`, `datasets`, `fsspec`, backend phù hợp với cloud storage.
- Evaluation: thư viện evaluator tracking tương ứng và các công cụ tính detection metrics.

Cần pin phiên bản dependency sau khi kiểm thử để đảm bảo tái lập giữa local và Colab.

---

## 10. Kế hoạch triển khai và checklist

### Phase 1 — Dataset

- [ ] Kiểm tra quyền truy cập và tải thử UA-DETRAC.
- [ ] Đọc cấu trúc ảnh, XML và trajectory annotation.
- [ ] Xây dựng parser và class mapping.
- [ ] Tạo split theo video sequence.
- [ ] Chuyển dataset thành shard và thử streaming trên runtime mới.
- [ ] Kiểm tra cache tạm, retry, checksum và xử lý lỗi tải.

### Phase 2 — Detection

- [ ] Chuẩn hóa dataset cho PyTorch.
- [ ] Huấn luyện Faster R-CNN từ random initialization.
- [ ] Chạy YOLOv8 pretrained baseline.
- [ ] Implement và thử R-CNN/Fast R-CNN.
- [ ] Implement và thử custom RT-DETR.
- [ ] Lưu checkpoint, config và detection metrics.

### Phase 3 — Tracking và Polygon Zones

- [ ] Xây dựng tracker interface.
- [ ] Tích hợp SORT và ByteTrack.
- [ ] Tích hợp BoT-SORT.
- [ ] Cấu hình polygon theo từng camera.
- [ ] Thử riêng vùng xa với threshold/resolution phù hợp.
- [ ] Đánh giá HOTA, IDF1, MOTA và ID switches.

### Phase 4 — Speed Estimation

- [ ] Xây dựng công cụ chọn điểm calibration.
- [ ] Tạo homography riêng cho từng video/camera.
- [ ] Chuyển bottom-center bbox sang world coordinates.
- [ ] Tính và làm mượt vận tốc theo timestamp/FPS.
- [ ] Kiểm tra kết quả với khoảng cách/vận tốc đo thực tế nếu có.

### Phase 5 — End-to-end Demo

- [ ] Kết nối detector + tracker + zones + speed estimator.
- [ ] Chạy video demo có sẵn.
- [ ] Xuất MP4 có bbox, class, ID, speed và polygon.
- [ ] Xuất CSV/JSON.
- [ ] Viết hướng dẫn chạy lại trên Colab/máy mới.
- [ ] Tổng hợp bảng so sánh model và tracker.

---


## 11. Git và quản lý dữ liệu

Không đưa dữ liệu lớn và output vào Git:

```gitignore
# Dataset and temporary cache
data/
*.zip
*.tar
*.tar.gz
*.mp4

# Training outputs
outputs/
checkpoints/
runs/
wandb/
tensorboard/

# Python
__pycache__/
*.py[cod]
.venv/
.env
.ipynb_checkpoints/
```

Các file cần version control:

- Source code, config YAML và script chuẩn bị dữ liệu.
- Manifest chứa tên shard, split và metadata (không chứa credential).
- README, requirements/pyproject và test.

Các file không commit:

- UA-DETRAC gốc hoặc bản đã chuyển đổi.
- Hugging Face token, cloud access key hoặc signed URL riêng tư.
- Checkpoint lớn, log và video output.

---

## 13. Tài liệu tham khảo

1. **UA-DETRAC project page:** <https://sites.google.com/view/daweidu/projects/ua-detrac>
2. Longyin Wen et al., *UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking*, Computer Vision and Image Understanding, 2020. <https://doi.org/10.1016/j.cviu.2020.102907>
3. **Ultralytics tracking documentation** (ByteTrack, BoT-SORT và các tracker khác): <https://docs.ultralytics.com/modes/track/>
4. **Supervision PolygonZone documentation:** <https://supervision.roboflow.com/latest/detection/tools/polygon_zone/>
5. **Torchvision detection models documentation:** <https://docs.pytorch.org/vision/stable/models.html>

---

## 14. Kết quả mong đợi

Một hệ thống modular có thể thay thế detector và tracker bằng config, train/evaluate trên UA-DETRAC, truy cập dữ liệu từ cloud/runtime mà không cần lưu dataset lâu dài trên máy cá nhân, và xuất video tracking kèm vận tốc sau khi camera được hiệu chuẩn.

**Trạng thái:** README mô tả yêu cầu, kiến trúc và kế hoạch triển khai của project. Các CLI command, module và config trong tài liệu là giao diện/định hướng cần được implement và kiểm thử trong quá trình phát triển; không có nghĩa toàn bộ chức năng đã tồn tại sẵn.
