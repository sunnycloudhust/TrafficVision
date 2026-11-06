# Kiến trúc và quy ước thí nghiệm

## Luồng dữ liệu

```text
XML + frames → parser → canonical JSONL / TAR shards
                             ↓
                 DatasetSource → PyTorch sample
                             ↓
                      detector adapter
                             ↓
         full frame + far-zone crops → NMS → zone thresholds
                             ↓
                      tracker adapter
                             ↓
          bottom-center → world projection → speed history
                             ↓
                    annotation + CSV/JSON
```

Dataset không biết model. Detector không biết tracker. Tracker không biết calibration. Speed estimator nhận ID, tọa độ mét, timestamp giây và Jacobian truyền nhiễu pixel sang hệ tọa độ thế giới. Pipeline điều phối các module này.

## Interface chung

- Input detector: RGB `FloatTensor[3,H,W]`, khoảng 0–1.
- Target detection: `boxes: float32[N,4]` theo `xyxy`, `labels: int64[N]` theo 1–4, `image_id: int64`, `track_ids`, `ignored_regions`, `area`, `iscrowd`.
- Output detector: `boxes`, `scores`, `labels` trên **kích thước ảnh input gốc**, tensor CPU qua adapter.
- Output tracker: `Track(track_id, box, label, score)`; chỉ trả track đang quan sát, lưu track mất trong nội bộ để reactivation.
- Tọa độ thế giới: mét. Vận tốc: km/h. Không có calibration thì giá trị speed/world là `null` trong JSON và ô rỗng trong CSV.
- Frame dataset one-based; timestamp frame đầu là 0. XML giữ origin tọa độ liên tục gốc, không ngầm thay đổi convention.

## Custom detectors

| Detector | Implementation | Khác biệt cần ghi khi báo cáo |
|---|---|---|
| Faster R-CNN | Backbone tự dựng hoặc ResNet18; AnchorGenerator, RPNHead, RegionProposalNetwork, RoIAlign, MLP và class/box predictor được lắp riêng | Dùng primitives Torchvision đã kiểm thử; không sao chép toàn bộ RPN/NMS/CUDA kernel. CNN mặc định nhỏ, không tương đương ResNet50-FPN benchmark |
| Fast R-CNN | Cùng shared image CNN/RoI heads, thay RPN bằng Selective Search | RoIAlign thay RoI Pooling; proposals xử lý CPU, không có learned RPN |
| R-CNN | Crop từng proposal từ ảnh, CNN trên từng crop, head classification/regression | Joint softmax head thay SVM pipeline gốc. Crop resize 64 pixel; chi phí tăng theo số proposal |
| RT-DETR educational | CNN đa tỉ lệ P3/P4/P5, attention trên scale cao, fusion hai chiều, IoU-aware classification, top-k encoder query, iterative decoder, Hungarian/L1/GIoU | Dùng dense attention, không có deformable attention/denoising và không tái lập đầy đủ uncertainty-minimal query selection. Không dùng để tuyên bố official RT-DETR accuracy/FPS |

`init=random` không tải pretrained weights. `imagenet` chỉ dành cho ResNet18 backbone của nhóm R-CNN. YOLO pretrained được quản lý riêng. Config/checkpoint ghi rõ init để tránh nhầm transfer learning với scratch.

Tham khảo kiến trúc gốc: [Torchvision detection components](https://docs.pytorch.org/vision/stable/models.html), [bài RT-DETR](https://arxiv.org/abs/2304.08069), [implementation chính thức](https://github.com/lyuwenyu/RT-DETR). Nhóm nên trình bày biến thể của mình cùng các thành phần đã lược bỏ.

## Ignored regions và đánh giá

Project dùng quy tắc **tâm bbox nằm trong ignore rectangle**. Faster R-CNN không lấy negative anchors trong vùng ignore; RoI training bỏ proposal ignore rồi thêm GT positives. RT-DETR bỏ classification negatives trong vùng ignore. YOLO stock loss không hỗ trợ ignored rectangles nên export ảnh mask vùng ignore, giữ lại pixel của GT positives.

Evaluator chung dùng ảnh gốc, loại prediction trong ignore trừ prediction khớp labeled positive cùng class ở IoU ≥ .5. Chính sách này phải được ghi trong báo cáo; không khẳng định tương đương evaluator UA-DETRAC chính thức. Native validation của YOLO trên ảnh export và validation chung có thể khác nhau; dùng evaluator chung để lập bảng so sánh cuối.

Detection AP dùng TorchMetrics/pycocotools. Precision/recall là greedy score-order matching một-một theo class ở IoU .5 và score .25. Tracking dùng [TrackEval chính thức](https://github.com/JonathonLuiten/TrackEval) pin commit `12c8791b303e0a0b50f753af204249e622d0281a`; adapter đưa trajectory arrays trực tiếp vào HOTA/CLEAR/Identity. Một view NumPy riêng cho hai metric cũ thay alias `np.float/np.int`; không sửa NumPy global. Kiểm thử có cả perfect matching và ID switch.

GT và prediction được remap ID liên tục trong từng video; IoU của hai class khác nhau bằng 0; tổng hợp bằng `combine_sequences` của evaluator. `track-dataset` yêu cầu frame đầy đủ, tăng dần từ 1 và reset tracker giữa sequence. HOTA/IDF1 không được tính bằng số ID đếm trên ảnh.

## Remote source

Canonical TAR đặt cặp `<key>.jpg`, `<key>.json` liền nhau. Reader không extract tên member ra filesystem và chỉ giữ một cặp sample trong bộ nhớ. Multiworker phân chia **theo shard**, không cho mỗi worker đọc lại toàn bộ split. Epoch shuffle thứ tự shard ở train; validation/test giữ thứ tự. Không hỗ trợ distributed/DDP hiện tại; train dùng một process CPU/GPU.

Cache ghi `.part`, verify rồi atomic replace. Stream kiểm tra SHA256 từng ảnh; cache kiểm tra thêm checksum toàn shard. Lỗi mạng trong cache được retry có giới hạn. Lỗi giữa shard ở mode stream được báo ra để người chạy retry từ checkpoint; không tự replay sample đã yield vì có thể làm trùng dữ liệu. Không gọi `snapshot_download` toàn dataset. HTTP local đã được kiểm tra; quyền HF/S3 và throughput thực là bước kiểm chứng của nhóm.

## Training và reproducibility

Custom loop dùng AdamW, StepLR, gradient clipping; config lưu seed, split manifest, init, kiến trúc và hyperparameters. Best checkpoint theo validation COCO mAP, last checkpoint dùng resume. RNG PyTorch/Python/DataLoader được lưu tại ranh giới epoch; GPU RNG được lưu khi có. Streaming workers được tạo lại mỗi epoch để nhận epoch seed. Chưa có AMP, DDP, augmentation pipeline phong phú hoặc khả năng resume giữa batch; đây là nền để nhóm mở rộng sau khi có baseline thực.

Không thể đảm bảo bitwise reproducibility giữa phần cứng/PyTorch khác nhau. Giữ `splits.json`, manifest, config, environment constraints và logs cho mọi kết quả báo cáo.

## Speed và timestamp

Homography kiểm tra finite, nonsingular, điểm trùng/thẳng hàng và fit error. Khi chiếu gần horizon, pipeline bỏ quan sát và reset history. Pipeline dùng Kalman constant-velocity 2D; độ lớn vector vận tốc nhân 3.6 ra km/h. Jacobian giải tích thay bốn phép chiếu sai phân; covariance cập nhật Joseph và gain tính bằng solve. Uncertainty dùng trị riêng lớn nhất của velocity covariance, có tính tương quan hai trục. Chuyển động rẽ gấp cần kiểm chứng với camera thực.

Gap dài hoặc timestamp không tăng khởi tạo lại trạng thái; innovation outlier trả N/A và reset sau `max_rejects`. Timeout tính từ observation cuối được chấp nhận, không từ lần predict/reject. `max_reasonable_kmh` và uncertainty gate kiểm soát việc xuất tốc độ, không chứng minh một track không có ID switch.

Video mặc định timestamp `frame_index/FPS`. Mode PTS đọc `CAP_PROP_POS_MSEC` và từ chối timestamp không đơn điệu. Output MP4 là constant-FPS theo nguồn, CSV giữ timestamp đã dùng để tính. FPS và frame size calibration phải khớp video, không tự scale homography mà không có xác nhận hình học.

Chỉ dùng calibration của mặt đường tương ứng. Ma trận homography của cảnh kiểm thử tổng hợp chỉ dùng để kiểm tra phần mềm, không áp dụng cho video thực. UA-DETRAC trajectory/bbox không phải GT vận tốc mét/giây.
