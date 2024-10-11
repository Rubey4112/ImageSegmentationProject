# pip install cyndilib opencv-python

import cv2
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.sender import Sender
from cyndilib.wrapper.ndi_structs import FourCC
from cyndilib.video_frame import VideoSendFrame
from fractions import Fraction

video = "./image_segmentation/SegmentationTestVid.mov"
# cv2.namedWindow("NDI Receiver", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("NDI Receiver", 640, 480)

cap = cv2.VideoCapture(video)


sender = Sender("Testing")

 # Build a VideoSendFrame and set its resolution and frame rate
    # to match the options argument
vf = VideoSendFrame()
vf.set_resolution(1280, 720)
fr = Fraction(30000, 1001)
vf.set_frame_rate(fr)
vf.set_fourcc(FourCC.BGRA)

# Add the VideoSendFrame to the sender
sender.set_video_frame(vf)

while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2G)
    image_bytes = memoryview(bytearray(cv2.imencode('.jpg', frame)[1].tobytes()))
    sender.write_video(image_bytes)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()