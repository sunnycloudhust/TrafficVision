# Review và sửa chữa sau cập nhật của leader

Ngày: 2026-11-06. Base: `main` tại commit `db13bd6`.
Branch: `fix/kalman-speed-and-review-tools`.

## Phạm vi review

Đã đọc các cập nhật về camera YAML, Kalman speed estimator, polygon labeler, progress bars, training và CLI. Giữ hướng Kalman và các giá trị calibration/hyperparameter mà nhóm đang dùng. Chưa train hoặc tuning lại detector.

## Những vấn đề đã sửa

| Vấn đề | Thay đổi | Tác dụng |
|---|---|---|
| Kalman nhận noise bằng 0, giá trị NaN hoặc Jacobian suy biến | Kiểm tra tham số ngay khi khởi tạo; kiểm tra shape/finite/rank Jacobian, reset track lỗi | Báo lỗi rõ ràng, tránh truyền dữ liệu hỏng vào phép giải hệ tuyến tính |
| Covariance cập nhật bằng dạng rút gọn; gain dùng inverse | Joseph form, đối xứng hóa covariance; dùng `solve` cho gain | Cải thiện ổn định số của Kalman |
| Uncertainty chỉ nhìn hai phần tử đường chéo | Dùng trị riêng lớn nhất của velocity covariance | Gate theo hướng bất định lớn nhất, tính cả tương quan hai trục |
| Observation bị reject vẫn cập nhật thời gian dùng cho timeout | Tách `observed_t` và thời gian predict | Chuỗi outlier không giữ trạng thái cũ sống quá thời gian cho phép |
| Mỗi track cần bốn phép chiếu để xấp xỉ Jacobian | Thêm `Homography.jacobian()` giải tích; helper gọi công thức này | Giảm chi phí tính và bỏ sai số do bước sai phân 0.5 pixel |
| GUI cho phép lưu polygon chưa hoàn tất, rỗng hoặc suy biến | Chặn save không hợp lệ; kiểm tra điểm phân biệt/diện tích/tên; clamp tọa độ click vào frame | YAML tạo ra dùng được với zone manager, giảm việc vô tình bỏ mất polygon đang vẽ |
| README/Colab gọi `scripts/utils.py` không tồn tại | Đổi sang `scripts/run_trafficvision.py` | Copy lệnh từ tài liệu có thể chạy được |
| Tài liệu còn mô tả regression speed cũ | Đồng bộ README và architecture với Kalman hiện tại | Người đọc hiểu đúng thuật toán và giới hạn |
| Test inheritance hard-code LR 0.0001 trong khi base đã là 0.001 | So sánh với config base | Kiểm tra đúng inheritance, không ép nhóm quay lại LR cũ |

Ngoài ra, progress bar video cập nhật postfix với `refresh=False` để không ép redraw mỗi frame. Sửa unused import, đánh dấu bootstrap import của script và chuẩn hóa format. Các thay đổi trong CLI, tracking metrics và trainer chỉ là format.

Helper `pixel_jacobian` hiện nhận `(homography, point)`; tham số bước sai phân `eps` của bản cũ không còn cần thiết. Các caller trong repo đã dùng dạng hai tham số.

## Kiểm chứng

Môi trường: Python 3.12.3, PyTorch 2.5.1+cpu, Torchvision 0.20.1+cpu. Dùng môi trường đã có ở workspace, nhưng import source của checkout mới; không dựa vào package cũ để chạy demo.

- Trước sửa: **35 passed, 1 failed**; lỗi là test LR hard-code. Ruff báo 9 lỗi.
- Sau sửa: **55 passed**. Thêm 19 case cho tham số/Jacobian lỗi, covariance symmetric/PSD, gap/timestamp trùng, reject timeout, derivative perspective và polygon labeler.
- `ruff check src scripts tests data-labelling --no-cache`: đạt.
- `ruff format --check --no-cache src scripts tests data-labelling`: 87 file đạt.
- `git diff --check`: đạt.
- `python scripts/run_trafficvision.py --help`: chạy được.
- Labeler được kiểm tra logic và YAML headless; chưa kiểm tra thao tác cửa sổ desktop trên WSL.

### Đo riêng Jacobian

Cùng một homography perspective và một điểm, chạy 5.000 lần mỗi cách, lấy thời gian nhỏ nhất trong 3 lượt:

| Cách tính | Thời gian |
|---|---|
| Sai phân cũ, bốn phép chiếu | 0.3246 giây |
| Jacobian giải tích | 0.0869 giây |

Khoảng **3,74 lần** cho riêng Jacobian trên CPU này. Không suy ra FPS toàn pipeline tăng cùng tỷ lệ; validation rank/eigenvalue và detector vẫn có chi phí riêng.

### Chạy lại video giao thông thật

Dùng lại clip `vehicles.mp4` từ [Supervision assets](https://supervision.roboflow.com/assets/), đã tải ở workspace; bản 720p dài 21,52 giây, 25 FPS. Chạy source mới với YOLOv8n COCO pretrained + ByteTrack, không dùng calibration của camera khác.

- Xử lý và decode lại đủ **538/538 frame**.
- CSV và JSON đều có **1455 observation**, **17 tracking IDs**.
- Khoảng **9,15 FPS** trên CPU trong lượt này. Không coi đây là phép so sánh tốc độ có kiểm soát với lượt chạy cũ.
- Không có calibration nên speed là `N/A`/null trong toàn bộ export.
- Artifacts local: `outputs/review/real_video.mp4`, `.csv`, `.json`, `.summary.json`, `validation.json`, `preview_400.jpg`; được gitignore.

Lượt video kiểm tra inference/tracking/export. Kalman đã được kiểm tra trên quỹ đạo có tốc độ biết trước trong test; clip thật không có GT/calibration để xác minh sai số km/h. Số ID/observation không phải số xe ground-truth hay tracking accuracy.

## Phần nhóm cần kiểm chứng tiếp

Train/fine-tune và chấm UA-DETRAC cùng split; xác minh khoảng cách đo thực và calibration trên điểm độc lập; chấm tốc độ khi có GT; thử GUI trên desktop và runtime CUDA. Không thay các ma trận camera hoặc tự bổ sung khoảng cách giả trong PR này.
