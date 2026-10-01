# Bottle Cap Vision

## 1. Mục tiêu

Bottle Cap Vision là đồ án Python/OpenCV dùng để nhận diện nắp chai và phân loại 10 màu. Hai luồng chính thức của Môn 1 là xử lý ảnh và video có sẵn. Luồng video bổ sung tracking, ổn định màu theo thời gian và đếm nắp khi đi qua đường đếm.

Tài liệu này dành cho giảng viên, sinh viên và người tiếp tục bảo trì dự án.

## 2. Trạng thái chức năng

| Chế độ | Trạng thái | Phạm vi chấp nhận cuối |
| --- | --- | --- |
| Image | Active, supported | **PASS / FINAL** |
| Video | Active, supported | **PASS / FINAL** |
| Camera | Implementation retained, development frozen | Không thuộc demo bắt buộc; chất lượng nhận diện thực tế được deferred |

Camera không bị xóa và vẫn dùng chung lõi xử lý tuần tự. Tuy nhiên dữ liệu hiện tại chưa đại diện đủ cho cảnh webcam/thiết bị đích, vì vậy không xem camera là chế độ production ngang với Image và Video.

## 3. Chức năng chính

### Image

- Đọc một ảnh hoặc thư mục ảnh.
- Nhận diện nắp chai bằng learned detector và phân loại màu.
- Vẽ kết quả, xuất ảnh chú thích, JSON, CSV và summary.

### Video

- Đọc video có sẵn và chạy learned inference tuần tự trên từng frame.
- Theo dõi đối tượng bằng `NearestNeighborTracker`.
- Ổn định màu theo track bằng `ColorStabilizer`.
- Đếm line crossing, tránh đếm lại cùng track và trả `RunSummary`.
- Hiển thị theo kiểu realtime, hỗ trợ dừng bằng `Q` hoặc `Esc`.

### Experimental / Frozen Camera Mode

- Implementation được giữ lại cho mục đích tham khảo và tái sử dụng.
- Runtime structure đã qua automated tests.
- Chất lượng nhận diện trong môi trường camera thực được deferred cho đến khi có dataset đúng target domain.

## 4. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Cài dependency dành cho test/phát triển:

```powershell
pip install -r requirements-dev.txt
```

Cài dependency training:

```powershell
pip install -r requirements-training.txt
```

Runtime production dùng `ultralytics==8.4.160`.

## 5. Lệnh chính thức

### Image

Chạy thư mục mặc định `data/input/images/` và mở giao diện:

```powershell
python scripts/demo_images.py
```

Chạy không mở GUI:

```powershell
python scripts/demo_images.py --no-gui
```

Các tham số hợp lệ khác là `--input`, `--output`, `--config` và `--recursive`. Output mặc định nằm tại `outputs/practical_demo/`.

### Video

```powershell
python scripts/demo_video.py "data\input\videos\bottle_cap_conveyor_v2.mp4"
```

Có thể dùng `--config <path>` hoặc `--max-frames <n>`. Không dùng `--max-frames` cho lần acceptance chính thức vì video phải chạy đến EOF.

### Launcher

```powershell
python scripts/run_practical.py
```

Menu vẫn hiển thị Image, Camera và Video. Image và Video là hai workflow được chấp nhận cuối; Camera chỉ là implementation được giữ lại.

### Camera thử nghiệm đã đóng băng

```powershell
python scripts/demo_camera.py
```

Các tham số hợp lệ là `--camera-index`, `--config` và `--max-frames`. Camera không bắt buộc trong demo cuối của Môn 1.

## 6. Kiến trúc cuối

Image flow:

```text
Image input
    -> LearnedDetector
    -> Detection results
    -> Annotation
    -> Image output / JSON / CSV
```

Video flow:

```text
VideoSource
    -> LearnedDetector
    -> NearestNeighborTracker
    -> ColorStabilizer
    -> LineCounter
    -> Display / output / RunSummary
```

Camera flow giữ nguyên cùng lõi tuần tự:

```text
CameraSource
    -> LearnedDetector
    -> NearestNeighborTracker
    -> ColorStabilizer
    -> LineCounter
    -> Display / RunSummary

Status: FROZEN / REAL-WORLD MODEL QUALITY DEFERRED
```

I/O nằm ngoài `BottleCapPipeline`; detection/tracking/stabilization/counting chạy trên dữ liệu trong bộ nhớ. Xem [kiến trúc cuối](docs/FINAL_ARCHITECTURE.md) để dùng lại khi viết báo cáo.

## 7. Production model

- Backend: Ultralytics learned detector.
- Model family: `yolo11n.pt`.
- Runtime: `ultralytics==8.4.160`.
- Bundle: `models/p9_full_001`.
- Checkpoint: `models/p9_full_001/checkpoints/best.pt`.
- SHA-256: `9A8300A78B076ACC33AAF81C9FA76BF3C12B9FE6FDEA792CE044C100164426FB`.
- Image size: `320`.
- Confidence threshold: `0.25`.
- NMS: class-agnostic (`agnostic_nms=True`).

Thứ tự 10 class canonical:

1. Blue
2. Brown
3. Green
4. Lilac
5. Orange
6. Pink
7. Purple
8. Red
9. White
10. Yellow

Runtime load trực tiếp từ stable bundle và không phụ thuộc `outputs/experiments`.

## 8. Dataset và training

- Format: YOLO object detection.
- Số class canonical: 10.
- Training được chấp nhận: P9 full, experiment `p9_full_001`, 50 epochs.
- Dataset fingerprint gắn với checkpoint cuối: `77a5489d41ef7312e2cde0a31b962696c19d9ac8f962d5fbd71a905c0d77dbc0`.

Dataset training không có supervised sample cho Lilac và chỉ có 6 object Yellow ở train; coverage của cảnh camera/thiết bị đích cũng còn yếu. Nếu triển khai với phần cứng thực, cần thu thập dữ liệu đúng camera, ánh sáng, nền và khoảng cách sử dụng rồi retrain.

## 9. Video acceptance

Video chuẩn: `data/input/videos/bottle_cap_conveyor_v2.mp4`.

| Thuộc tính | Kết quả |
| --- | --- |
| Kích thước | 1280×720 |
| Source FPS | 15 |
| Số frame | 120 |
| Số vật thể vật lý | 8 |
| Expected crossings | 8 |
| Final count | 8 |

Class-agnostic NMS đã đưa số frame có cross-class duplicate từ 34 xuống 0, loại duplicate track của object 5 và sửa count từ 9 về 8 mà vẫn giữ được các nắp khác nhau nằm gần nhau. Source/frame accounting, learned inference, tracking cardinality, chống recount cùng track và EOF flow đều đã được chấp nhận.

Một nắp White vật lý có thể dao động Blue/White ở raw detector. Đây là giới hạn model/dataset, không phải lỗi tracking hoặc counting; vì vậy acceptance hạ tầng video không bị ràng buộc vào một phân bố màu tuyệt đối.

## 10. Cấu trúc repository

```text
bottle_cap_vision/
configs/
data/
  input/
    images/
    videos/
  dataset/
  external_staging/
models/
  p9_full_001/
docs/
scripts/
tests/
outputs/
  p12/
  p9_full_quality/
README.md
CHANGELOG.md
requirements.txt
requirements-dev.txt
requirements-training.txt
pyproject.toml
```

Các thư mục cache và output tạm không thuộc cấu trúc release.

## 11. Release status và giới hạn đã biết

Image và Video đã đạt acceptance kỹ thuật. Camera được giữ lại nhưng đóng băng; không mô tả camera là hỏng và cũng không tuyên bố chất lượng nhận diện camera thực đã PASS.

Các asset bắt buộc hiện có trong working tree nhưng đang untracked trong Git: production model bundle, canonical video và demo image. Khi tạo bản nộp, phải thêm rõ các asset này vào archive; Git-only archive hiện chưa đủ. Trạng thái gate chi tiết nằm tại [docs/FINAL_STATUS.md](docs/FINAL_STATUS.md).
