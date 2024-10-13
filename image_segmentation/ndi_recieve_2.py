# pip install cyndilib opencv-python

import cv2
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync
from ultralytics import YOLO
import numpy as np
import torch

model = YOLO("yolo11m-seg.pt")

cap = cv2.VideoCapture(cv2.CAP_DSHOW)

finder = Finder()
# Create a Receiver without a source
receiver = Receiver(
    color_format=RecvColorFormat.RGBX_RGBA,
    bandwidth=RecvBandwidth.highest,
)
source = None
video_frame = VideoFrameSync()

# Add the video/audio frames to the receiver's FrameSync
receiver.frame_sync.set_video_frame(video_frame)

def on_finder_change():
    global source
    if finder is None:
        return
    ndi_source_names = finder.get_source_names()
    if len(ndi_source_names) == 0:
        return
    if source is not None:
        # already playing a source
        return
    print("Setting source to", ndi_source_names[0]) ####### Change sources here
    with finder.notify:
        source = finder.get_source(ndi_source_names[0])
        receiver.set_source(source)

finder.set_change_callback(on_finder_change)
finder.open()

cv2.namedWindow("NDI Receiver", cv2.WINDOW_NORMAL)
cv2.resizeWindow("NDI Receiver", 640, 480)

while True:
    success, frame = cap.read()
    if receiver.is_connected():
        receiver.frame_sync.capture_video()

        if min(video_frame.xres, video_frame.yres) != 0:
            ndi_frame = video_frame.get_array() 
            ndi_frame = ndi_frame.reshape(video_frame.yres, video_frame.xres, 4)[:,:,:3]
            ndi_frame = cv2.cvtColor(ndi_frame, cv2.COLOR_RGBA2BGR)

            results = model(frame)
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
            
            
            isolated = cv2.bitwise_and(mask, ndi_frame)
            img = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
            
            # Display the frame
            cv2.imshow("NDI Receiver", isolated)

    # Check for a key press
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        break

# Clean up
cv2.destroyAllWindows()
if receiver.is_connected():
    receiver.disconnect()
finder.close()