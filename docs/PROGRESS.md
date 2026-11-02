# Báo cáo triển khai theo phase

Cập nhật: **2026-11-02**. Workspace ban đầu chỉ có tài liệu yêu cầu, chưa có code, Git repo, dataset hoặc checkpoint.

**Trạng thái bàn giao:** đã triển khai nền tảng và chạy kiểm thử chức năng cho cả 5 phase. Chưa hoàn tất thực nghiệm trên UA-DETRAC hoặc xác minh vận tốc ngoài thực tế. Không dùng số đo synthetic làm kết quả của môn học.

## Phase 1 — Dataset

Đã làm:

- Parser XML UA-DETRAC, mapping class 1–4, bbox xywh→xyxy, frame one-based, trajectory IDs, attributes và ignored regions.
- Dò layout ảnh theo sequence thay vì giả định archive cố định; báo lỗi ảnh thiếu/trùng, class lạ và ID trùng.
- Split tái lập theo video, cho phép cung cấp split official qua JSON.
- Local JSONL, TAR shards, shard manifest chứa SHA256/size/count; uploader HF mặc định private và nhận token qua môi trường.
- Local/HF/cloud source; `stream`, `cache`, `auto`, cache atomic/retry/checksum và phân shard cho DataLoader workers.

Đã kiểm tra: fixtures XML/ảnh qua preprocessing→shard→dataset; ba chế độ đọc; checksum hỏng; video leakage; HTTP streaming/cache qua server local; hai workers không đọc trùng sample.

Còn cần dữ liệu/quyền truy cập: tải một sequence và XML gốc, overlay để xác nhận coordinate convention và mapping của bản tải; upload repo HF thực rồi thử trên runtime mới. [Trang tác giả](https://sites.google.com/view/daweidu/projects/ua-detrac) vẫn liệt kê nguồn tải; chưa tải archive ảnh lớn hay xác nhận quota Google Drive bằng file thực.

## Phase 2 — Detection

Đã làm:

- Dataset PyTorch trả tensor/target thống nhất; custom detector registry tách YOLO baseline.
- Faster R-CNN tự lắp backbone/RPN/RoI/head từ primitives Torchvision; random init mặc định.
- Fast R-CNN dùng shared CNN + Selective Search; R-CNN chạy CNN riêng từng crop.
- RT-DETR educational với multi-scale CNN/hybrid encoder/query selection/decoder/Hungarian loss.
- YOLOv8 pretrained adapter, kiểm tra tên class, remap COCO rõ ràng, export cùng split và native fine-tuning entry point.
- AdamW/StepLR/gradient clipping; validation mAP; config/logs; best/last checkpoint; epoch resume với RNG.
- Evaluator chung cho AP/precision/recall/latency/model parameter size/CUDA memory.

Đã kiểm tra: 4 model forward/loss/backward trên ảnh có nhãn và nhãn rỗng; tọa độ output ở kích thước ảnh gốc; cả 4 model train một epoch nhỏ, lưu/nạp checkpoint và resume epoch kế tiếp; Selective Search chạy thật; export YOLO đúng class offset/split. Đã tải YOLOv8n pretrained và chạy video 5 frame qua ByteTrack, tạo MP4.

Còn cần thực nghiệm: train scratch/fine-tune trên UA-DETRAC đầy đủ, test GPU/CUDA, chọn checkpoint theo validation rồi báo cáo test. Chưa có mAP/FPS benchmark của dữ liệu thật. Chưa chạy fine-tune YOLO đầy đủ.

Giới hạn kiến trúc cần trình bày: R-CNN dùng joint softmax head, không tái lập SVM pipeline gốc; RT-DETR rút gọn dense attention, chưa phải implementation đầy đủ của bài báo. Chi tiết ở [ARCHITECTURE.md](ARCHITECTURE.md).

## Phase 3 — Tracking và Polygon Zones

Đã làm:

- SORT Kalman/Hungarian IoU; adapters ByteTrack/BoT-SORT cho mọi detector.
- Association riêng theo class, track buffer, chỉ xuất track đang có observation.
- Polygon bottom-center, threshold theo vùng, crop inference vùng xa và map box về ảnh gốc.
- `track-dataset` chạy toàn split, reset giữa video, export trajectory và evaluate bằng GT IDs.
- TrackEval HOTA/IDF1/MOTA/ID switches/fragments theo protocol project.

Đã kiểm tra: SORT hồi phục qua gap và hết hạn; low-confidence association và reactivation cho ByteTrack/BoT-SORT; polygon boundary; crop mapping; quỹ đạo perfect được metric 1 và cố ý ID switch làm metric giảm.

Còn cần thực nghiệm: polygon đúng camera thực, đánh giá recall vùng xa và benchmark cả ba tracker trên cùng split. BoT-SORT hiện tắt ReID; adapter báo lỗi nếu cố bật khi chưa tích hợp encoder. Chưa có bảng HOTA/IDF1/MOTA của UA-DETRAC.

## Phase 4 — Speed Estimation

Đã làm:

- Công cụ calibration từ JSON point pairs hoặc chọn điểm GUI desktop.
- Homography kiểm tra degeneracy, singularity, fit error và horizon.
- Bottom-center→mét; hồi quy vị trí theo timestamp trong cửa sổ.
- Kiểm tra số quan sát tối thiểu, timestamp lỗi, gap dài, ID jump/vận tốc bất thường; history cleanup.
- Speed evaluator MAE/RMSE/MAPE từ cặp GT/prediction đã đối chiếu.

Đã kiểm tra: phép chiếu với tọa độ mét đã biết; quỹ đạo tốc độ 36 km/h với timestamp không đều; timestamp trùng, gap dài, bước nhảy; GT bằng 0 khi tính MAPE; không có calibration thì export speed/world là null.

Còn cần đo thực: điểm đường có khoảng cách đã biết, calibration riêng camera và GT speed nếu muốn báo sai số km/h. Camera mẫu để `homography.matrix: null`; không tạo ma trận giả cho video thật.

## Phase 5 — End-to-end Demo

Đã làm:

- CLI train/evaluate/demo/track-dataset/evaluate-tracking/evaluate-speed/smoke.
- Video pipeline, lựa chọn FPS/PTS, kiểm tra độ phân giải/FPS/camera ID, đóng tài nguyên khi có lỗi.
- MP4 overlay bbox/class/ID/speed/zones; CSV và JSON ghi từng observation, summary runtime.
- Synthetic smoke có sẵn, notebook kiểm tra annotation, tài liệu local/Colab, script tổng hợp report thành comparison CSV và CI CPU.

Đã kiểm tra: ghi/đọc lại MP4 đủ frame; CSV/JSON cùng số observation; tracking ID ổn định; cảnh synthetic 50 frame, 25 FPS, 0.1 m/pixel và 2 pixel/frame cho kết quả khoảng **18 km/h**. Đã chạy video pipeline với YOLOv8n pretrained, chưa có video giao thông thực của nhóm.

Artifacts xem ngay trong workspace:

- `outputs/smoke/synthetic_input.mp4`
- `outputs/smoke/synthetic_demo.mp4`, `.csv`, `.json`, `.summary.json`
- `outputs/smoke/checks.json`
- `outputs/smoke/yolo_pretrained.mp4` và exports tương ứng

Artifacts nằm trong `.gitignore`; clone mới tạo lại bằng `python -m trafficvision.cli smoke`.

## Kết quả kiểm tra môi trường

- Python 3.12.3, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu, NumPy 1.26.4.
- Bộ kiểm thử: **35 passed** trên CPU.
- Ruff lint và format đã kiểm tra; package cài được qua `pip install -e ".[train,tracking-eval,dev]"`.
- TrackEval pin commit chính thức; không phụ thuộc một phiên bản PyPI có requirement NumPy xung đột môi trường này.
- CI đã viết, chưa chạy trên GitHub vì workspace chưa có remote Git.
- `requirements/constraints-tested.txt` ghi snapshot dependency đã kiểm thử.

## Bảng kết quả thực nghiệm chờ điền

| Thí nghiệm | mAP .5 | mAP .5:.95 | HOTA | IDF1 | MOTA | FPS | Speed MAE |
|---|---|---|---|---|---|---|---|
| Faster R-CNN scratch + ByteTrack | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa đo | Cần GT |
| YOLOv8n fine-tune + ByteTrack | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa đo | Cần GT |
| Fast R-CNN + SORT | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa đo | Cần GT |
| R-CNN educational + SORT | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa đo | Cần GT |
| RT-DETR educational + BoT-SORT | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy | Chưa đo | Cần GT |

Thứ tự làm tiếp của nhóm: xác minh dataset thật → train Faster R-CNN và YOLO baseline → chấm cùng split → chọn video/camera → calibration → demo → so sánh các model/tracker còn lại. Các giá trị cấu hình hiện tại phục vụ khởi chạy, chưa được tuning.
