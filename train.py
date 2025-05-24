from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='data.yaml',
    epochs=5,
    imgsz=640,
    batch=1,
    device='cpu'
)
