from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11m-seg.pt")

# Run inference on 'bus.jpg' with arguments
for r in model.predict("day _theater.mp4", save=True, imgsz=(720,1280), stream=True):
    pass