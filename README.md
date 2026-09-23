# BottleCapVision Python Base — P10 Learned Inference Boundary

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

Runtime video giữ frame index 0-based xuyên suốt `FramePacket`, `FrameResult`, CSV
và count event. `VideoSource` không tự bịa FPS: nếu container không cung cấp FPS hợp
lệ thì source để metadata là `None`, và `VideoSink` fail rõ ràng vì không thể tạo
video đúng tốc độ. Output video dùng kích thước và FPS của source, codec mặc định
`mp4v`, đồng thời từ chối output trùng input. JSON cuối run có `frames_processed`
để đối chiếu trực tiếp với số frame decode từ video và số record CSV.

Checkpoint smoke P9/P10 có cấu hình runtime riêng, không thay đổi default HSV:

```powershell
python main.py --source data/input/videos/sample.mp4 `
  --config configs/config.learned-p9-smoke.json `
  --output outputs/learned_result.mp4 `
  --csv outputs/learned_frames.csv `
  --json outputs/learned_result.json `
  --no-display --max-frames 50
```

`average_processing_fps` được tính bằng số frame chia tổng thời gian chỉ bao quanh
`core.process_frame`; nó không phải source FPS và không bao gồm model load hay sink I/O.

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

## Evaluation boundary P8

`bottle_cap_vision.evaluation` đánh giá mọi prediction source bằng cùng một contract
canonical: `sample_id`, canonical `class_id`, bbox normalized P7 và `score` optional.
Evaluator match bbox one-to-one trước khi so class, vì vậy bbox đúng nhưng sai màu
được giữ là class confusion thay vì bị tách thành localization miss. Confusion matrix
mở rộng có sentinel `BACKGROUND` cho missed GT và false positive.

```python
from bottle_cap_vision.dataset import DatasetManifest
from bottle_cap_vision.evaluation import EvaluationPrediction, evaluate_predictions

manifest = DatasetManifest.load("outputs/dataset_manifest.json")
predictions: list[EvaluationPrediction] = []  # producer/adapter supplies these
report = evaluate_predictions(manifest, predictions, split="val", iou_threshold=0.5)
report.save("outputs/evaluation_report.json")
```

Metric không xác định được serialize thành JSON `null`; macro average ghi rõ số class
được include. AP/mAP chưa được tính vì HSV `color_ratio` không phải confidence đã
calibrate. Evaluator không đọc lại raw YOLO labels, ảnh, tracker hoặc counter.

## Training boundary P9

`bottle_cap_vision.training` chỉ nhận canonical `DatasetManifest`. Adapter tạo một
dataset view riêng trong experiment bằng hard-link ảnh và label sinh từ canonical
annotations; chỉ `train` và `val` được materialize. Raw dataset và split `test` không
bị sửa hoặc dùng để optimize/chọn checkpoint.

Cài backend ML riêng (HSV-only runtime không phụ thuộc ML stack):

```powershell
pip install -r requirements-training.txt
```

Preflight không chạy optimization:

```powershell
python scripts/train_model.py `
  --manifest outputs/dataset_manifest.json `
  --dataset-root data/dataset/archive `
  --config configs/training_smoke.json `
  --dry-run
```

Bỏ `--dry-run` để chạy smoke. Config mẫu giới hạn 12 train/4 val, 1 epoch và mọi
artifact đều mang cờ `smoke_checkpoint_not_final_model`. Xóa hai trường
`train_sample_limit`/`val_sample_limit`, đổi experiment name và cấu hình compute để
train toàn bộ canonical train/val qua cùng code path. Experiment name đã tồn tại sẽ
bị từ chối, không overwrite.

Ultralytics có thể tải base checkpoint `yolo11n.pt` và font ở lần chạy đầu; model ID,
backend/version, effective arguments, seed, device, dataset fingerprint, exact class
mapping và canonical `best`/`last` checkpoint đều được ghi vào metadata. Sau khi asset
đã có local, training không cần discovery hoặc download dataset qua mạng. Seeded plan
và backend hỗ trợ configuration/data reproducibility; không cam kết bitwise identical
cho mọi GPU/kernel.

## Learned inference boundary P10

Runtime chọn backend một lần tại composition root. `HSVDetector` và
`LearnedDetector` đều trả `list[Detection]`; tracker, color stabilizer, counter và sink
không biết backend đang chạy. Learned adapter xác minh checkpoint metadata với
experiment metadata, canonical P7 manifest snapshot và class names nhúng trong model
trước inference.

```json
"detector": {
  "backend": "learned",
  "checkpoint": "outputs/experiments/p9_smoke_001/checkpoints/best.pt",
  "metadata": "outputs/experiments/p9_smoke_001/checkpoints/checkpoint_metadata.json",
  "device": "cpu",
  "confidence_threshold": 0.25
}
```

Ultralytics giữ quyền sở hữu preprocessing/NMS/letterbox reversal. Adapter chỉ
nhận result đã trở về original-frame coordinates và chuyển `xyxy` sang
canonical `(x, y, width, height)`. `Detection.confidence` là model score của frame
hiện tại, không phải temporal vote confidence của P5. Checkpoint smoke P9
không được xem là model production-ready.

## Known baseline limitation

Color classification chưa phải phiên bản calibration cuối. Một số dải HSV starter có thể chồng lấn ở các cặp gần nhau như Red/Pink, Purple/Lilac hoặc Orange/Brown. Đây là phần chủ động để lại cho giai đoạn **single-image color calibration** bằng dataset thật; không nên "fix" bằng hardcode dựa trên một ảnh mẫu duy nhất.
