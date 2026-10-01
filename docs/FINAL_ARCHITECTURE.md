# Final Architecture

## Mục tiêu

Kiến trúc cuối của Môn 1 tách I/O khỏi lõi xử lý ảnh. Image và Video là hai workflow chính thức; Camera dùng lại cùng lõi nhưng được giữ ở trạng thái frozen/deferred.

## Image workflow

```text
Image input
    -> LearnedDetector
    -> Detection results
    -> Annotation
    -> Annotated image + predictions.json + predictions.csv + summary.json
```

Image runner đọc từng file độc lập. Detector load từ production bundle, kết quả được chú thích và xuất qua lớp practical demo; không cần tracker hoặc counter xuyên nhiều ảnh.

## Video workflow

```text
VideoSource
    -> FramePacket
    -> LearnedDetector
    -> NearestNeighborTracker
    -> ColorStabilizer
    -> LineCounter
    -> FrameResult
    -> Display / RunSummary
```

- `VideoSource` giữ frame index và source FPS từ file.
- `LearnedDetector` trả detection theo tọa độ frame gốc; Ultralytics sở hữu preprocessing và class-agnostic NMS.
- `NearestNeighborTracker` duy trì identity giữa các frame.
- `ColorStabilizer` ổn định màu theo observation mới của track.
- `LineCounter` phát event khi track vượt đường đếm và không đếm lại cùng identity.
- Runtime sở hữu vòng lặp, timing, stop request và cleanup; `BottleCapPipeline` chỉ xử lý dữ liệu trong bộ nhớ.

## Camera workflow

```text
CameraSource
    -> same sequential core as Video
    -> Display / RunSummary
```

Implementation camera được giữ lại và runtime structure đã PASS. Development được đóng băng; nhận diện thực tế được deferred vì dataset hiện tại thiếu camera-domain coverage. Camera không thuộc final required demo.

## Model và dữ liệu

- Production checkpoint: `models/p9_full_001/checkpoints/best.pt`
- Model family: `yolo11n.pt`
- Ultralytics: `8.4.160`
- Input size: `320`
- Dataset: YOLO object detection, 10 class
- Dataset fingerprint: `77a5489d41ef7312e2cde0a31b962696c19d9ac8f962d5fbd71a905c0d77dbc0`

Runtime bundle tự chứa checkpoint metadata, experiment metadata, training config và manifest snapshot cần để kiểm tra compatibility; inference không phụ thuộc `outputs/experiments`.

## Input và output

| Workflow | Input | Output chính |
| --- | --- | --- |
| Image | Ảnh hoặc thư mục ảnh | Annotated images, JSON, CSV, summary |
| Video | File video | GUI display, `RunSummary` |
| Camera | Camera index | GUI display, `RunSummary` |

## Giới hạn đã biết

Model có thể nhầm hoặc dao động màu khi màu/ánh sáng mơ hồ. Canonical video vẫn đạt đúng 8 identity/count; chất lượng phân lớp màu không được dùng để che hoặc thay thế contract tracking/counting. Việc dùng camera ngoài môi trường lab cần target-domain data và retraining.
