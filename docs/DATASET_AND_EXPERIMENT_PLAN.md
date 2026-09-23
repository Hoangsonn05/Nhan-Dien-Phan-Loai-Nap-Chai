# Dataset and Experiment Plan v1.1

## Dataset reference
Dataset hiện được dùng làm nguồn kiểm thử / hiệu chỉnh cho bài toán màu nắp chai. Nếu dataset YOLO có annotation sẵn, annotation có thể dùng cho ground truth mà không bắt buộc kiến trúc runtime phải là YOLO.

## Audit trước khi calibration
1. kiểm tra cây thư mục;
2. đếm ảnh / object theo class;
3. xác thực label nếu có;
4. tìm label rỗng hoặc malformed;
5. kiểm tra duplicate / near-duplicate;
6. thống kê resolution và object size;
7. tạo sample grid theo từng màu.

## Split
Ưu tiên train/reference - validation - blind test sao cho không rò rỉ các frame gần giống nhau giữa các tập.

## Experiments
- E1: Python/OpenCV HSV baseline trên ảnh.
- E2: threshold/geometry calibration trên validation set.
- E3: video + tracking + line crossing.
- E4: color stabilization qua nhiều frame.
- E5: chỉ khi baseline chưa đạt yêu cầu, đánh giá learned detector/classifier.

## Metrics
- color classification accuracy;
- macro-F1 / per-class recall nếu phù hợp;
- confusion matrix;
- detection success / false positive / false negative;
- total count accuracy;
- count error by color;
- processing time / FPS.
