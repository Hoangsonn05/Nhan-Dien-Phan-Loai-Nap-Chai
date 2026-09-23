# Python/OpenCV Core Flow v1.1

## Input
Python tự chịu trách nhiệm acquisition:
- ảnh: `cv2.imread()`;
- video: `cv2.VideoCapture()`;
- webcam có thể bổ sung sau bằng `cv2.VideoCapture(0)` nếu scope thay đổi.

Không cần phần mềm trung gian để cấp ảnh/video.

## Baseline pipeline

```text
BGR frame
 -> ROI
 -> HSV
 -> color masks
 -> morphology
 -> contours
 -> area/circularity/solidity filtering
 -> color candidate
 -> centroid/bbox
 -> tracker
 -> crossing line
 -> count event
 -> visualization / logging
```

## Configuration rule
`configs/config.json` là file runtime cấu hình. Nó phải có danh sách màu hợp lệ. Explicit `"colors": []` được coi là config lỗi và loader sẽ raise `ValueError`.

## Current assumptions
- camera/video tương đối cố định;
- nắp có kích thước pixel nằm trong giới hạn cấu hình;
- màu được biểu diễn đủ rõ trong HSV;
- tracker hiện tại là nearest-neighbor baseline.

## Development order
1. config + project integrity;
2. single-image color detection;
3. dataset calibration / evaluation;
4. video detection;
5. tracking stability;
6. color stabilization;
7. line-crossing count;
8. reporting/UI nếu cần.
