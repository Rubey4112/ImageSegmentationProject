# pip install cyndilib opencv-python

import cv2
from cyndilib.wrapper.ndi_recv import RecvColorFormat, RecvBandwidth
from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync

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
    if receiver.is_connected():
        receiver.frame_sync.capture_video()

        if min(video_frame.xres, video_frame.yres) != 0:
            # Convert the frame to a format that OpenCV can use
            frame = video_frame.get_array() 
            frame = frame.reshape(video_frame.yres, video_frame.xres, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            # Display the frame
            cv2.imshow("NDI Receiver", frame)

    # Check for a key press
    key = cv2.waitKey(30)
    if key == ord("q") or key == 27:
        break

# Clean up
cv2.destroyAllWindows()
if receiver.is_connected():
    receiver.disconnect()
finder.close()