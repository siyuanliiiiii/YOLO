from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/home/ml164/YOLO/runs/detect/train9/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("/home/ml164/YOLO/original.jpg", save=True, imgsz=640, conf=0.5)

results = model()