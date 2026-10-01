# Final Status

## Release decision

**STATUS: NOT READY TO FREEZE**

Baseline kỹ thuật của Môn 1 đã PASS cho Image và Video. Freeze/release archive chưa được tuyên bố hoàn tất vì production model bundle, canonical video và demo image vẫn untracked trong Git. Các asset đã có trong working tree nhưng phải được thêm rõ vào bản nộp; một Git-only archive hiện sẽ thiếu chúng.

## Modes

| Mode | Implementation | Final acceptance | Status |
| --- | --- | --- | --- |
| Image | PASS | PASS | FINAL |
| Video | PASS | PASS | FINAL |
| Camera | PASS | Not required | RETAINED / FROZEN |

Camera runtime structure đã qua automated tests. Chất lượng nhận diện camera thực được **DEFERRED**, không phải FAIL, vì dataset hiện chưa đại diện đủ target domain.

## Core

| Thành phần | Trạng thái |
| --- | --- |
| Config | PASS |
| Learned detector | PASS |
| Tracking | PASS |
| Color stabilization contract | PASS |
| Counter | PASS |
| Runtime I/O | PASS |

## Model

- Bundle: `models/p9_full_001`
- Checkpoint: `models/p9_full_001/checkpoints/best.pt`
- SHA-256: `9A8300A78B076ACC33AAF81C9FA76BF3C12B9FE6FDEA792CE044C100164426FB`
- Backend/model: Ultralytics `8.4.160`, `yolo11n.pt`
- Image size: `320`
- Confidence threshold: `0.25`
- NMS: `agnostic_nms=True`
- Dataset fingerprint: `77a5489d41ef7312e2cde0a31b962696c19d9ac8f962d5fbd71a905c0d77dbc0`
- Classes: Blue, Brown, Green, Lilac, Orange, Pink, Purple, Red, White, Yellow
- Runtime dependency on `outputs/experiments`: **NO**

## Release assets

| Asset | Path | Git state |
| --- | --- | --- |
| Production model | `models/p9_full_001/` | UNTRACKED |
| Canonical video | `data/input/videos/bottle_cap_conveyor_v2.mp4` | UNTRACKED |
| Demo image | `data/input/images/WIN_20210818_16_10_24_Pro_jpg.rf.3271fd13498c23c5b267bf05368d1ecd.jpg` | UNTRACKED |

Các path trên là danh sách bắt buộc khi tạo submission/source archive. Task này không tự chạy `git add` hoặc commit.

## Final automated regression

- Full suite: **302 tests PASS** bằng `.venv\Scripts\python.exe -m pytest -q`.
- `compileall`: **PASS** với `python -m compileall -q bottle_cap_vision scripts tests`.
- `git diff --check`: **PASS**.
- Cache residue ngoài `.venv`: **0** (`__pycache__`, `.pytest_cache`, `*.pyc`).

Lần chạy đầu bằng `pytest` ngoài venv thất bại do interpreter đó thiếu `torch` và `ultralytics`; chạy đúng project venv với Ultralytics `8.4.160` đã PASS toàn bộ 302 tests.

## Practical acceptance

### Image

- Command: `.\.venv\Scripts\python.exe scripts\demo_images.py --no-gui`
- Processed: 1 image
- Failed: 0
- Detections: 52
- Result: **PASS**

### Video

- File: `data/input/videos/bottle_cap_conveyor_v2.mp4`
- Frames processed: 120
- Source FPS: 15.00
- Average processing FPS: 37.68
- Physical objects / expected crossings: 8 / 8
- Total count: 8
- Count by color: Purple 2, Red 2, Orange 1, Green 2, Blue 1
- EOF/no traceback: PASS
- Result: **PASS**

Phân bố màu không phải release gate. Một nắp White vật lý có thể dao động Blue/White ở detector thô do chất lượng model/data.

### User stop

- Automated runtime stop/cleanup coverage: PASS trong test suite.
- Phím `Q`/`Esc` trên GUI thực: **PENDING MANUAL USER VERIFICATION**.

### Launcher và Camera

- Launcher menu smoke/clean exit: PASS.
- Image/Video dispatch contract: PASS trong automated tests.
- Camera retained in launcher: YES.
- `python scripts/demo_camera.py --help`: PASS.
- Webcam test: Not required.

## Known limitations

- Camera-domain generalization chưa đủ tin cậy vì thiếu dữ liệu đúng môi trường triển khai.
- Phân loại màu có thể dao động ở các màu mơ hồ; đây không phải lỗi counting/tracking.
- Dataset P9 không có supervised train sample cho Lilac và chỉ có 6 object Yellow ở train.
- Triển khai phần cứng tương lai cần thu thập target-domain data và retrain.

## Freeze condition

Không cần phát triển thêm thuật toán cho Môn 1. Sau khi xác nhận production model, canonical video và demo image đã nằm trong submission/archive thực tế, baseline có thể đổi thành `STATUS: FROZEN` mà không cần sửa core algorithm.

Môn 2 có thể tái sử dụng frame-processing core hiện có và tích hợp hardware I/O sau; thiết kế Môn 2 nằm ngoài phạm vi tài liệu này.
