import cv2, time
from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image

model = YOLO("yolov8m-seg.pt")
# model = FastSAM("FastSAM-s.pt")

video = "SegmentationTestVid.mov"
cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if success:
        start = time.perf_counter()
        results = model.predict(frame)
        # for result in results:
        result = results[0]
        # get array results
        masks = result.masks.data
        boxes = result.boxes.data
        # extract classes
        clss = boxes[:, 5]
        # get indices of results where class is 0 (people in COCO)
        people_indices = torch.where(clss == 0)
        # use these indices to extract the relevant masks
        people_masks = masks[people_indices]
        # scale for visualizing results
        people_mask = torch.any(people_masks, dim=0).int() * 255
        # save to filef
        # cv2.imwrite('./merged_segs.jpg', people_mask.cpu().numpy())
        # color_converted = cv2.cvtColor(people_mask.cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(people_mask.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask, frame)
        cv2.imshow("YOLOv8 Inference", isolated)

        # end = time.pernd - start
        # fps = 1 / totf_counter()
        # total_time = eal_time

        # annotated_frame = results[0].plot()

        # cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        # cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()




# img= cv2.imread('/usr/local/lib/python3.10/dist-packages/ultralytics/assets/zidane.jpg')
# model = YOLO('yolov8n-seg.pt')
# results = model.predict(source=img.copy(), save=True, save_txt=False, stream=True)
# for result in results:
#     # get array results
#     masks = result.masks.data
#     boxes = result.boxes.data
#     # extract classes
#     clss = boxes[:, 5]
#     # get indices of results where class is 0 (people in COCO)
#     people_indices = torch.where(clss == 0)
#     # use these indices to extract the relevant masks
#     people_masks = masks[people_indices]
#     # scale for visualizing results
#     people_mask = torch.any(people_masks, dim=0).int() * 255
#     # save to file
#     cv2.imwrite(str(model.predictor.save_dir / 'merged_segs.jpg'), people_mask.cpu().numpy())