# Dataset and Experiment Plan v2.0

## Dataset reference
Dataset tại `data/dataset/archive/` là bộ object-detection được export từ Roboflow theo YOLOv5. Mỗi annotation có dạng `class_id x_center y_center width height`; bốn tọa độ được normalized. Một ảnh có thể có nhiều object. File label rỗng là negative sample hợp lệ theo policy P7.

Mapping chính thức từ `data.yaml`:

```text
0 blue, 1 brown, 2 green, 3 lilac, 4 orange,
5 pink, 6 purple, 7 red, 8 white, 9 yellow
```

Tên dataset lowercase khác casing với tên runtime. P7 ghi warning và giữ nguyên mapping dataset; không rename hoặc alias ngầm.

## Audit trước khi calibration
1. kiểm tra cây thư mục;
2. đếm ảnh / object theo class;
3. xác thực label nếu có;
4. tìm label rỗng hoặc malformed;
5. kiểm tra exact duplicate và cross-split leakage bằng SHA-256;
6. thống kê resolution và object size;
7. sinh statistics và deterministic manifest.

## Split
Split vật lý hiện có là `train`, `valid`, `test`; Dataset Layer map explicit `valid -> val` và không reshuffle. P7 chỉ kiểm tra exact duplicate; near-duplicate/perceptual similarity nằm ngoài phạm vi.

## Manifest contract
`DatasetManifest` schema `1.0` chứa mapping, ordered samples, normalized bbox, metadata ảnh, content hash, split/class statistics và semantic fingerprint. Paths là POSIX-style relative paths từ dataset root. P8/P9 phải load manifest này thay vì parse raw folder hoặc label.

## Experiments
- E1: Python/OpenCV HSV baseline trên ảnh.
- E2: threshold/geometry calibration trên validation set.
- E3: video + tracking + line crossing.
- E4: color stabilization qua nhiều frame.
- E5: chỉ khi baseline chưa đạt yêu cầu, đánh giá learned detector/classifier.

## Metrics
- P8 dùng `EvaluationPrediction(sample_id, class_id, bbox, score?)` với bbox normalized
  cùng semantics P7; score optional chỉ là metadata finite, không tự coi `color_ratio`
  là probability;
- localization match one-to-one bằng greedy IoU giảm dần (mặc định 0.5), rồi mới
  so class; bbox đúng nhưng sai class tạo confusion, FN cho GT class và FP cho
  predicted class;
- precision/recall không xác định được biểu diễn bằng `null`; F1 chỉ có khi cả
  precision và recall được định nghĩa; macro average bỏ qua riêng các metric
  `null` và report số class được include;
- confusion matrix mở rộng dùng GT rows, prediction columns và `BACKGROUND` để
  biểu diễn missed GT / unmatched prediction;
- AP/mAP defer cho đến khi prediction source có ranking confidence có ý nghĩa;
- total count accuracy;
- count error by color;
- processing time / FPS.
