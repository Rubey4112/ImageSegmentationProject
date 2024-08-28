import cv2, time
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")
# model = FastSAM("FastSAM-s.pt")

video = "SegmentationTestVid.mov"
model.predict(video, save = True, show=True)
# cap = cv2.VideoCapture(video)

# while cap.isOpened():

#     success, frame = cap.read()

#     if success:
#         start = time.perf_counter()
#         results = model(frame, save = True)

#         end = time.perf_counter()
#         total_time = end - start
#         fps = 1 / total_time

#         annotated_frame = results[0].plot()

#         cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()