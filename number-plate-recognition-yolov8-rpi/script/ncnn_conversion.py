from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("../models/yolov8n.pt")

# Export the model to NCNN format
#model.export(format="ncnn", imgsz=640)  # creates '..._ncnn_model'
model.export(format="ncnn", imgsz=320)
