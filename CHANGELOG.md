# Changelog

## Unreleased — P8 Evaluation boundary

- Thêm canonical prediction contract dùng class ID và normalized bbox từ P7.
- Thêm deterministic one-to-one IoU matching, tách localization correspondence khỏi class correctness.
- Báo cáo TP/FP/FN, per-class/micro/macro metrics và extended confusion matrix có `BACKGROUND`.
- Undefined metrics dùng `null`; zero-support class không sinh recall giả.
- Thêm deterministic JSON report, validation và regression/smoke tests trên manifest thật.
- Defer AP/mAP cho đến khi prediction source có confidence ranking có ý nghĩa.

## v1.4.0 — P7 Dataset boundary

- Xác nhận dataset thật là Roboflow YOLOv5 object detection với split `train/valid/test`.
- Thêm canonical sample/annotation contract, strict audit, class statistics và negative-sample policy explicit.
- Phát hiện pairing errors, corrupt image, malformed bbox, duplicate content và cross-split leakage.
- Thêm deterministic portable manifest, semantic fingerprint và load API cho P8/P9.
- Giữ nguyên mapping lowercase từ `data.yaml`; chỉ cảnh báo casing mismatch với runtime.
- Không thay đổi detector, tracker, stabilizer, counter hoặc runtime I/O.

## v1.3.0 — P4 Runtime I/O separation

- Tách `ImageSource` và `VideoSource` khỏi vision core.
- Chuẩn hóa `FramePacket`, `FrameResult`, `RuntimeMetrics` và `RunSummary`.
- Giới hạn `BottleCapPipeline` vào detection/tracking/counting trên frame trong bộ nhớ.
- Chuyển annotation, display, image/video, CSV và JSON output sang các sink độc lập.
- Thêm runtime runner sở hữu timing, `max_frames`, stop request và cleanup.
- Chuyển `main.py` và `scripts/run_image.py` sang cùng Source → Core → Sink flow.
- Giữ nguyên detector, tracker, counter và config semantics P0–P3.

## v1.2.0 — P0 Foundation clean

- Thêm `bottle_cap_vision/models.py` làm canonical runtime contract cho `Detection`, `Track`, `CountEvent`.
- Loại duplicate dataclass definitions khỏi detector/tracker/counter; giữ compatibility import cũ.
- Gom annotation vào `BottleCapPipeline.annotate_frame()`.
- Thêm `BottleCapPipeline.run_image()` làm đường xử lý ảnh duy nhất cho `main.py` và `scripts/run_image.py`.
- Giảm logic OpenCV/visualization bị lặp trong hai CLI entry points.
- Chuẩn hóa package exports và version thành `1.2.0`.
- Bổ sung regression test cho shared contracts và canonical image pipeline.
- Bổ sung `.gitkeep` cho thư mục data rỗng để cấu trúc không mất khi đóng gói.
- Không thay đổi thuật toán association của tracker, line-crossing, HSV thresholds hoặc config semantics; các phần này thuộc P1–P3/phases sau.

## v1.1.0 — Foundation normalization

- Chuyển kiến trúc thành Python/OpenCV standalone.
- Loại DobotVisionStudio khỏi dependency và tài liệu kiến trúc chính.
- Chuẩn hóa package thành `bottle_cap_vision/`.
- Tách `configs/`, `data/`, `outputs/`, `tests/`, `docs/`, `scripts/`.
- Sửa lỗi `colors: []` làm vô hiệu hóa toàn bộ color detector.
- Bổ sung validation cho cấu hình màu/HSV.
- Bổ sung canonical `config.json` với đủ 10 màu.
- Sửa import của CLI và helper script.
- Bổ sung test config round-trip và regression test cho lỗi empty colors.
- Siết test line-crossing để chứng minh count đúng một lần.
- Bổ sung `.gitignore`, `requirements-dev.txt`, `pyproject.toml` và tài liệu Python-only.
