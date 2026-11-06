# Review GUI calibration và các sửa chữa

Ngày 2026-11-06. Base `main`: `32aa407`. Branch: `fix/calibration-gui-validation`.

## GUI thực sự làm gì?

`data/data-labelling/calibration_tool.py` chọn điểm ảnh và dựng tọa độ mét từ
**số đo thực do người dùng cung cấp**. Có hai chế độ: hình chữ nhật có dài/rộng
đã biết, hoặc hình khác có khoảng cách của mọi cặp điểm. Điểm bổ sung của hình
chữ nhật được xác định từ khoảng cách đến ba góc đầu.

GUI xuất JSON image/world points; `scripts/calibrate_camera.py` tạo camera YAML
chứa homography. Pipeline sau đó mới dùng homography, tracking và timestamp để
tính vận tốc. GUI không tự biết đường rộng/dài bao nhiêu từ một ảnh.

## Lỗi được xác định và đã sửa

| Vấn đề trên bản main | Sửa trong PR |
|---|---|
| Đổi chiều dài/rộng sau khi tính vẫn giữ world points cũ để save | Trace các input; hủy kết quả ngay khi sửa; so sánh snapshot input ở bước save |
| Pairwise MDS vẫn trả tọa độ khi các khoảng cách không nhất quán | Tái dựng khoảng cách từ tọa độ và từ chối nếu vượt tolerance |
| Trilateration chỉ báo residual, không chặn dữ liệu mâu thuẫn | Kiểm tra ba khoảng cách gốc trước khi nhận điểm; kiểm tra cả các khoảng cách bổ sung |
| Bốn góc rectangle có thể click chéo hoặc tạo hình lõm | Bắt buộc tứ giác lồi, đúng thứ tự quanh hình; kiểm tra điểm trùng, thẳng hàng, NaN |
| Chưa kiểm tra toàn bộ điểm có khớp một homography | Fit và kiểm tra reprojection tính bằng mét; chặn vùng calibration cắt qua horizon |
| Text khoảng cách đang nhập có thể bị bỏ qua khi solve/save | Hủy solution khi sửa text; yêu cầu lưu khoảng cách hoặc xóa text trước khi tính |
| Clear điểm vẫn có thể giữ tên cặp điểm cũ | Xóa selection cũ, kiểm tra indices trước khi ghi khoảng cách |
| Có thể click vùng đệm và tạo điểm ngoài ảnh mà không chủ đích | Mặc định chỉ nhận điểm trong ảnh; ngoại suy cần checkbox riêng; kiểm tra lại khi solve |
| Nhãn `--done` chỉ dựa vào độ dài hai list JSON | Kiểm tra dữ liệu numeric, shape, degeneracy, homography; metadata hỏng cũng không đánh done |
| Đường dẫn frame/output cố định | Bổ sung `--frames-dir` và `--output-dir` |

Thêm file provenance `calibration_points_<CAMERA>.json`: số đo, kích thước
rectangle, ngưỡng lỗi, frame, kích thước ảnh và danh sách điểm ngoại suy. Hai file
image/world JSON vẫn tương thích CLI hiện có. Ngưỡng mặc định **0.25 m** có thể
chỉnh trong GUI; đây là kiểm tra tính nhất quán đầu vào, không phải chứng nhận
sai số calibration hay vận tốc thực.

Phần toán được tách vào `src/trafficvision/geometry/calibration_points.py`,
không phụ thuộc Tk. Các tọa độ camera/point annotation đã commit của nhóm không
bị chỉnh sửa. Sửa thêm một lỗi lint E402 ở CLI bootstrap; zone labeler chỉ đổi format.

## Kiểm chứng đã chạy

- Main trước sửa: **55 tests passed**; lint lỗi E402 ở `scripts/run_trafficvision.py`.
- Sau sửa: **78 tests passed**, gồm 23 case mới.
- Geometry tests: giữ khoảng cách khi dựng MDS, reject khoảng cách vô lý/cặp trùng,
  trilateration, thứ tự góc, điểm bổ sung, fit homography và ngưỡng lỗi.
- Tk tests đã chạy thật qua display WSL hiện có, không skip: tạo GUI bằng
  `tk.Tk()` với cửa sổ ẩn, dùng callback thật để solve/save/edit/clear và kiểm tra
  dữ liệu viết ra. Các messagebox được thay bằng collector để không chặn test.
- Test toàn luồng: GUI lưu JSON → gọi CLI calibration trong process riêng →
  đọc camera YAML → chiếu một điểm biết trước, kết quả **[5, 10] mét** trong
  sai số 1e-6 của cảnh kiểm thử có hình học đã biết.
- Ruff lint đạt; format check đạt **91 file**; `git diff --check` đạt.
- `python data/data-labelling/calibration_tool.py --help` chạy được.

Các lệnh kiểm tra:

```bash
python -m pytest -q
ruff check src scripts tests data/data-labelling --no-cache
ruff format --check --no-cache src scripts tests data/data-labelling
```

Nếu không có display, các test Tk được skip, còn geometry tests vẫn chạy.
Muốn chạy Tk tests trên Linux headless có thể dùng `xvfb-run -a python -m pytest`.
Chưa kiểm tra thao tác chuột thủ công của mọi loại màn hình/DPI.

## Điểm cần nhóm thống nhất khi dùng video quay chéo/gần ngang

- Chỉ chọn rectangle khi **trên mặt đường thực sự có rectangle có kích thước đã
  đo**, không biến một tứ giác bất kỳ thành đường rộng/dài theo số tùy đoán.
- Điểm ngoài ảnh chỉ dùng khi tọa độ chiếu của chúng đã được xác định độc lập.
  Click vào vùng xám không tạo thêm thông tin quan sát.
- Một fit bốn điểm có thể có reprojection gần 0 dù kích thước thực bị nhập sai.
  Cần kiểm chứng bằng các điểm/khoảng cách độc lập và cùng mặt phẳng đường.
- Nhãn `--done` nghĩa là bộ file qua kiểm tra số học, không phải đã xác minh
  accuracy tốc độ. Không có số đo/calibration đủ tin cậy thì giữ speed N/A.

PR này chưa đo MAE/RMSE vận tốc trên video thật, chưa thay đổi camera YAML hoặc
chạy tuning detector. Những việc đó cần số đo thực/GT của camera tương ứng.
