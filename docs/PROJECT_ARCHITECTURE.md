# Project Architecture v1.3 — P4 Source/Core/Sink

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

## 4. Source of truth
- Input: file ảnh/video đọc trực tiếp bằng OpenCV.
- Configuration: `configs/config.json`.
- Runtime code: package `bottle_cap_vision/`.
- Verification: `tests/` và dataset thực tế.

## 5. Module ownership
- `config.py`: schema, default values, JSON load/save/validation.
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
- Threshold phải nằm trong config, không rải trong nhiều module.
- Không thêm deep learning cho đến khi baseline được đo và evidence cho thấy cần thiết.

## 7. Known future work
- calibration HSV bằng dataset thật;
- xử lý overlap giữa các dải màu gần nhau;
- tách tracking identity khỏi color decision;
- color stabilization qua nhiều frame;
- đánh giá confusion matrix / accuracy / counting error;
- chỉ cân nhắc YOLO/classifier học máy nếu classical baseline không đạt tiêu chí.
