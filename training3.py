from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/ml164/YOLO/Dataset_3/data3.yaml", epochs=1000, scale=0.5, batch=8, degrees=30, multi_scale=True)