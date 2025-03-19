import cv2, time
from ultralytics import YOLO
import numpy as np
import torch
import NDIlib as ndi
import sys

model = YOLO("yolo11m-seg.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Camera resolution:", width, "x", height)

#### NDI ####
if not ndi.initialize():
    sys.exit(1)

#### NDI RECV ####
ndi_find = ndi.find_create_v2()

if ndi_find is None:
    sys.exit(1)

sources = []
while not len(sources) > 0:
    print('Looking for sources ...')
    ndi.find_wait_for_sources(ndi_find, 1000)
    sources = ndi.find_get_current_sources(ndi_find)
print([s.ndi_name for s in sources])

ndi_recv_create = ndi.RecvCreateV3()
ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA

ndi_recv = ndi.recv_create_v3(ndi_recv_create)

if ndi_recv is None:
    sys.exit(1)

ndi.recv_connect(ndi_recv, sources[0])
ndi.find_destroy(ndi_find)


#### NDI SEND ####
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'ndi-python'

ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()

ndi_recv_buffer = []
nframe = np.zeros((640,480,3),dtype=np.uint8)
while cap.isOpened():

    success, frame = cap.read()

    t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)

    if t == ndi.FRAME_TYPE_VIDEO:
        print('Video data received (%dx%d).' % (v.xres, v.yres))
        ndi_frame = np.copy(v.data)
        ndi_frame = cv2.resize(ndi_frame, (640,480))[:,:,:3]
        ndi_recv_buffer.append(ndi_frame)
        # cv2.imshow('ndi image', ndi_frame)
        ndi.recv_free_video_v2(ndi_recv, v)
    
    if success:
        start = time.perf_counter()
        results = model(frame)
        # for result in results:
        result = results[0]
        # print(result)
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
        if len(ndi_recv_buffer):
            nframe = ndi_recv_buffer.pop()     
        # print(nframe)  
        
        isolated = cv2.bitwise_and(mask, nframe)
        img = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
        video_frame.data = img
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

        ndi.send_send_video_v2(ndi_send, video_frame)
        cv2.imshow("YOLOv11 Inference", isolated)
        # ndi.recv_free_video_v2(ndi_recv, v)
        # end = time.pernd - start
        # fps = 1 / totf_counter()
        # total_time = eal_time

        # annotated_frame = results[0].plot()

        # cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        # cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            ndi.send_destroy(ndi_send)
            ndi.recv_destroy(ndi_recv)
            ndi.destroy()
            break
    # else:
    #     ndi.send_destroy(ndi_send)
    #     ndi.recv_destroy(ndi_recv)
    #     ndi.destroy()
    #     break

ndi.send_destroy(ndi_send)
ndi.recv_destroy(ndi_recv)
ndi.destroy()

cap.release()
cv2.destroyAllWindows()