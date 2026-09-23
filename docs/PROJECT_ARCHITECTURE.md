# Project Architecture v1.4 — P7 Runtime and Dataset Boundaries

## 1. Scope
Dự án hoạt động hoàn toàn trên phần mềm. Nguồn vào hiện tại là ảnh tĩnh hoặc video có sẵn. Không yêu cầu camera vật lý, PLC, robot, GPIO, băng tải thật hoặc DobotVisionStudio.

## 2. Objective
- phát hiện ứng viên nắp chai;
- phân loại màu;
- theo dõi đối tượng trong video;
- đếm duy nhất khi đối tượng qua vạch logic;
- thống kê tổng và theo màu;
- xuất ảnh/video annotate, CSV và JSON khi cần.

## 3. System boundary

```text
ImageSource / VideoSource
        |
        v
FramePacket (frame + frame_index)
        |
        v
BottleCapPipeline core
(detection -> tracking -> counting)
        |
        v
FrameResult
        |
        v
Runtime runner
        |
        v
Image / Video / Display / CSV / JSON sinks
```

Offline dataset flow is independent from runtime inference:

```text
Raw YOLO Dataset
        |
        v
Dataset discovery + strict audit
        |
        v
Canonical DatasetSample / Annotation records
        |
        v
Deterministic JSON manifest
        |
        v
Canonical P8 evaluator (matching / metrics / confusion matrix)
        |
        v
P9 training and P10 learned inference consumers
```

## 4. Source of truth
- Input: file ảnh/video đọc trực tiếp bằng OpenCV.
- Configuration: `configs/config.json`.
- Runtime code: package `bottle_cap_vision/`.
- Verification: `tests/` và dataset thực tế.

## 5. Module ownership
- `config.py`: schema, default values, JSON load/save/validation.
- `dataset.py`: YOLO discovery, validation, class mapping, leakage audit, statistics and manifest boundary.
- `evaluation.py`: model-agnostic prediction contract, deterministic one-to-one
  IoU matching, class-aware metrics and extended confusion matrix.
- `training.py`: validated config/preflight, deterministic manifest subset, backend adapter,
  non-destructive experiment metadata and canonical checkpoint contract.
- `models.py`: shared algorithm and runtime contracts (`Detection`, `Track`, `CountEvent`, `FramePacket`, `FrameResult`, `RunSummary`).
- `detector.py`: HSV masking, morphology, contours, geometry filter, classification candidate.
- `tracker.py`: identity tracking baseline.
- `counter.py`: crossing event và aggregation.
- `pipeline.py`: in-memory vision core; không sở hữu file, capture, writer hoặc GUI.
- `sources.py`: image/video acquisition và source metadata.
- `sinks.py`: annotation, display, image/video/CSV/JSON output.
- `runtime.py`: iteration, timing, stop, `max_frames` và cleanup.
- `visualization.py`: pure drawing/presentation helpers.
- `main.py`: CLI composition root.

## 6. Design constraints
- Không đếm detection theo từng frame.
- Một object chỉ tạo một count event.
- Không hardcode dataset path trong thuật toán.
- Evaluation/training phải consume canonical manifest, không rediscover hoặc parse raw labels riêng.
- Manifest chỉ dùng relative paths và loader từ chối path escape dataset root.
- Training chỉ materialize `train`/`val`; `test` không tham gia optimization hoặc model selection.
- Framework-specific training nằm sau adapter; runtime pipeline không load checkpoint trong P9.
- Class mapping giữ nguyên mapping chính thức từ `data.yaml`; không phụ thuộc filesystem order.
- Threshold phải nằm trong config, không rải trong nhiều module.
- P9 chỉ huấn luyện offline; learned checkpoint chưa được nối vào runtime pipeline trước P10.

## 7. Known future work
- calibration HSV bằng dataset thật;
- xử lý overlap giữa các dải màu gần nhau;
- tách tracking identity khỏi color decision;
- color stabilization qua nhiều frame;
- đưa HSV baseline và learned prediction sau này qua cùng P8 evaluator;
- đánh giá learned detector qua P8 sau khi P10 cung cấp canonical inference adapter.
