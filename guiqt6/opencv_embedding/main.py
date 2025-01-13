import sys
import cv2
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QTimer, QObject, Signal, QThread, Qt

from time import strftime, localtime

app = QGuiApplication(sys.argv)

engine = QQmlApplicationEngine()
engine.quit.connect(app.quit)
engine.addImportPath(sys.path[0])
engine.loadFromModule("Main", "Main")

class Backend(QThread):

    updated = Signal(str, arguments=['time'])
    updatedFrame = Signal(QImage, arguments=['frame'])
    updateFrame = Signal(QImage)


    def __init__(self):
        super().__init__()

        # Define timer.
        # self.timer = QTimer()
        # self.timer.setInterval(100)  # msecs 100 = 1/10th sec
        # self.timer.timeout.connect(self.update_time)
        # self.timer.start()
        self.status = True

    def update_time(self):
        # Pass the current time to QML.
        curr_time = strftime("%H:%M:%S", localtime())
        self.updated.emit(curr_time)

    def run(self):
        cam = cv2.VideoCapture(cv2.CAP_DSHOW)

        # Get the default frame width and height
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

        while self.status:
            ret, frame = cam.read()

            # Write the frame to the output file
            # out.write(frame)

            # Display the captured frame

            # Press 'q' to exit the loop
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             # Creating and scaling QImage
            h, w, ch = color_frame.shape
            img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)
        sys.exit(-1)



# Define our backend object, which we pass to QML.
backend = Backend()

engine.rootObjects()[0].setProperty('backend', backend)

# Initial call to trigger first update. Must be after the setProperty to connect signals.
backend.update_time()

sys.exit(app.exec())