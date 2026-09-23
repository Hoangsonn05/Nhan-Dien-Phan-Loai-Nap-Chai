# BottleCapVision Python Base v1.4 — P7 Dataset Boundary

Nền tảng software-only cho đồ án **nhận diện / phân biệt màu nắp chai**.

## Kiến trúc chính

```text
ImageSource / VideoSource
    -> FramePacket
    -> BottleCapPipeline (in-memory vision core)
       -> detection -> tracking -> counting -> FrameResult
    -> runtime runner
    -> Image / Video / Display / CSV / JSON sinks
```

DobotVisionStudio không còn là dependency của project. Python/OpenCV trực tiếp đọc ảnh và video.

## Cấu trúc

```text
BottleCapVision_Python_Base/
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── config.json
├── bottle_cap_vision/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── detector.py
│   ├── tracker.py
│   ├── counter.py
│   ├── pipeline.py
│   ├── sources.py
│   ├── sinks.py
│   ├── runtime.py
│   └── visualization.py
├── scripts/
│   └── run_image.py
├── tests/
│   ├── test_config.py
│   ├── test_core.py
│   ├── test_dataset.py
│   └── test_runtime.py
├── docs/
│   ├── PROJECT_ARCHITECTURE.md
│   ├── PYTHON_OPENCV_CORE_FLOW.md
│   └── DATASET_AND_EXPERIMENT_PLAN.md
├── data/
│   ├── input/images/
│   ├── input/videos/
│   └── dataset/
└── outputs/
```

## Cài đặt trên Windows / PyCharm terminal

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Kiểm tra nền tảng

```powershell
pytest
```

## Chạy một ảnh

Đặt ảnh vào `data/input/images/`, sau đó:

```powershell
python main.py --source data/input/images/sample.jpg --output outputs/image_result.jpg
```

hoặc:

```powershell
python scripts/run_image.py data/input/images/sample.jpg
```

## Chạy video

```powershell
python main.py --source data/input/videos/sample.mp4 --output outputs/result.mp4 --csv outputs/frames.csv --json outputs/result.json
```

Không mở cửa sổ preview:

```powershell
python main.py --source data/input/videos/sample.mp4 --output outputs/result.mp4 --no-display
```

## 10 màu cấu hình mặc định

- Blue
- Brown
- Green
- Lilac
- Orange
- Pink
- Purple
- Red
- White
- Yellow

`configs/config.json` đã chứa đầy đủ 10 màu. Các ngưỡng HSV hiện tại chỉ là baseline khởi đầu và cần hiệu chỉnh bằng dataset / video thực tế.

## Lỗi cấu hình đã sửa

Project cũ có:

```json
"colors": []
```

khiến `load_config()` ghi đè toàn bộ màu mặc định và detector không còn lớp màu để kiểm tra.

Bản này sửa ở hai tầng:

1. `configs/config.json` chứa đầy đủ 10 màu.
2. `load_config()` từ chối `colors: []` thay vì âm thầm làm detector mất toàn bộ màu. Nếu khóa `colors` không có, loader giữ cấu hình mặc định an toàn.

## Boundary runtime P4

`BottleCapPipeline.process_frame()` chỉ nhận frame trong bộ nhớ và trả `FrameResult`. Đọc image/video thuộc Source; annotation, display và ghi file thuộc Sink; timing, `max_frames`, stop request và cleanup thuộc runner. Cả `main.py` và `scripts/run_image.py` đều compose cùng boundary này. Thuật toán HSV, tracking association, line-crossing và config semantics P0–P3 không đổi.

## Dataset boundary P7

Dataset thật là Roboflow YOLOv5 object detection tại `data/dataset/archive/`, với split vật lý `train`, `valid` và `test`. `valid` được canonical hóa thành `val`. `data.yaml` là source of truth cho class ID và tên class; Dataset Layer giữ nguyên tên lowercase và không alias ngầm sang tên runtime Title Case.

Audit strict và build manifest:

```powershell
python -m bottle_cap_vision.dataset data/dataset/archive `
  --runtime-config configs/config.json `
  --audit-report outputs/dataset_audit.json `
  --manifest outputs/dataset_manifest.json
```

Label rỗng được biểu diễn là negative sample hợp lệ theo policy mặc định. Dùng `--require-objects` nếu contract của một dataset khác cấm negative sample. Manifest chỉ được ghi khi không có audit ERROR; nó chứa annotations canonical, relative paths, statistics, SHA-256 từng ảnh và fingerprint semantic để P8/P9 không parse raw YOLO lại.

Consumer API:

```python
from bottle_cap_vision.dataset import DatasetManifest

dataset = DatasetManifest.load("outputs/dataset_manifest.json")
for sample in dataset.split("train"):
    image_path = dataset.resolve_image("data/dataset/archive", sample)
```

## Known baseline limitation

Color classification chưa phải phiên bản calibration cuối. Một số dải HSV starter có thể chồng lấn ở các cặp gần nhau như Red/Pink, Purple/Lilac hoặc Orange/Brown. Đây là phần chủ động để lại cho giai đoạn **single-image color calibration** bằng dataset thật; không nên "fix" bằng hardcode dựa trên một ảnh mẫu duy nhất.
