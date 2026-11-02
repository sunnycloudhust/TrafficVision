# Chạy trên Colab hoặc máy mới

Hướng dẫn triển khai; **chưa chạy trực tiếp trên Colab/GPU trong workspace này**. Kiểm tra Python 3.10–3.12, GPU runtime và dung lượng runtime trước khi tải dữ liệu.

## 1. Clone và môi trường

```python
!git clone YOUR_REPOSITORY_URL /content/trafficvision
%cd /content/trafficvision
```

Chọn cặp torch/torchvision phù hợp runtime. Project đã kiểm tra CPU với `torch==2.5.1`, `torchvision==0.20.1`. Wheel CUDA 12.4 của cặp này được liệt kê trong [hướng dẫn phiên bản PyTorch](https://pytorch.org/get-started/previous-versions/):

```python
%pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
%pip install -e ".[train,tracking-eval,dev]"
%pip install --force-reinstall --no-deps opencv-contrib-python-headless==4.10.0.84
```

Nếu đã import torch/cv2 trước khi thay version, restart runtime rồi quay lại thư mục repo. Colab có thể cài sẵn torchaudio; nếu nó yêu cầu torch khác, gỡ hoặc cài torchaudio cùng phiên bản (project không dùng audio).

```python
import torch, torchvision, cv2
print(torch.__version__, torchvision.__version__, torch.cuda.is_available())
print(cv2.__version__, hasattr(cv2, "ximgproc"))
!python -m trafficvision.cli smoke --output outputs/smoke
```

Smoke cố định chạy CPU nhẹ, dù runtime có GPU. Train thật mới dùng `training.device: cuda`.

## 2. Nạp manifest remote

Dataset cần được một thành viên chuẩn bị/upload một lần bằng script trong README. Không có sẵn một HF repo được bịa trong code. Cấu hình `HF_TOKEN` qua Colab Secrets nếu repo private:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

from huggingface_hub import hf_hub_download
manifest = hf_hub_download(
    repo_id="YOUR_ORG/ua-detrac-shards", repo_type="dataset",
    filename="shards.json", local_dir="/content/trafficvision-manifest",
)
```

Chỉ file manifest nhỏ được tải ở bước này; shard URL trong manifest trỏ về HF. Không mount Google Drive cá nhân.

```python
from trafficvision.config import load_config
from pathlib import Path
import yaml

cfg = load_config("configs/experiments/faster_rcnn_scratch.yaml")
cfg["data"].update(source="huggingface", manifest=manifest, mode="cache",
                   cache_dir="/content/trafficvision-cache")
cfg["training"].update(device="cuda", workers=2, batch_size=2,
                       output="/content/trafficvision/outputs/faster_rcnn")
Path("outputs").mkdir(exist_ok=True)
Path("outputs/colab.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
```

Cache tải shard của split đang được sử dụng, chưa bảo đảm toàn bộ dataset vừa đĩa Colab. Kiểm tra dung lượng; nếu cần dùng `mode="stream"` hoặc tạo subset theo **video** và lưu manifest subset riêng. Không đổi test set theo kết quả chạy.

## 3. Train, evaluate, demo

```python
!python -m trafficvision.cli train --config outputs/colab.yaml
!python -m trafficvision.cli evaluate --model faster_rcnn --checkpoint outputs/faster_rcnn/best.pt --config outputs/colab.yaml --device cuda --split test --output outputs/evaluation/detection.json
!python -m trafficvision.cli track-dataset --model faster_rcnn --checkpoint outputs/faster_rcnn/best.pt --config outputs/colab.yaml --device cuda --tracker bytetrack --output outputs/evaluation/tracks.csv
```

Với video riêng, tải lên runtime hoặc truyền đường dẫn trực tiếp được OpenCV hỗ trợ rồi chạy `demo` như README. Chọn calibration points bằng JSON; Colab không hỗ trợ cửa sổ click OpenCV desktop.

```python
from IPython.display import Video
Video("outputs/smoke/synthetic_demo.mp4", embed=True)
```

Một số trình duyệt không phát codec `mp4v`; có thể tải MP4 về xem bằng VLC hoặc dùng ffmpeg đổi sang H.264. Việc decode/ghi file đã được kiểm tra bằng OpenCV; khả năng phát inline phụ thuộc codec của trình duyệt.

## 4. Giữ kết quả trước khi runtime kết thúc

Dataset cache tạm có thể bỏ; checkpoint/config/splits/metrics cần được nhóm lưu vào nơi đã chọn. Hướng dẫn này không tự upload lên dịch vụ bên ngoài. Có thể tải một file nhỏ bằng Colab:

```python
from google.colab import files
files.download("outputs/faster_rcnn/best.pt")
```

Chạy tiếp bằng `train --resume .../last.pt` sau khi nạp lại cùng config và split. Đừng dùng filesystem runtime làm nơi lưu checkpoint duy nhất của thí nghiệm dài.
