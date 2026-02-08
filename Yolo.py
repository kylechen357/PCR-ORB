# Using the ultralytics package
from ultralytics import YOLO

# Load a YOLOv8 segmentation model
model = YOLO('yolov8s-seg.pt')

# Export the model to TorchScript format with compatibility settings
model.export(format='torchscript', 
             imgsz=640,
             optimize=False)  # Disable optimizations that cause problems