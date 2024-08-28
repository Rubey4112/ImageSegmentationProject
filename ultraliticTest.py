import ultralytics
model = ultralytics.YOLO('yolov8n')
results = model.predict('./testImage.jpg', show=True)