# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause
from __future__ import annotations

import os
import sys
import time
import torch
import NDIlib as ndi
from ultralytics import YOLO
import numpy as np

import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QCheckBox)


"""This example uses the video from a  webcam to apply pattern
detection from the OpenCV module. e.g.: face, eyes, body, etc."""


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)

        ########## Class Field ###########
        self.trained_file = None
        self.status = True
        self.cap = True
        self.camera_preview = False
        self.model = YOLO("yolo11m-seg.pt")

        self.res_x = 640
        self.res_y = 480
        
        ####### End of Class Field ########
        

        #### NDI RECV ####
        self.ndi_find = ndi.find_create_v2()

        if self.ndi_find is None:
            sys.exit(1)

        # self.sources = []

        # print('Looking for sources ...')
        # while not len(self.sources) > 0:
        #     print(ndi.find_wait_for_sources(self.ndi_find, 1000))
        #     self.sources = ndi.find_get_current_sources(self.ndi_find)
        # print([s.ndi_name for s in self.sources])

        self.ndi_recv_create = ndi.RecvCreateV3()
        self.ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA

        self.ndi_recv = ndi.recv_create_v3(self.ndi_recv_create)

        if self.ndi_recv is None:
            sys.exit(1)

        ndi.find_destroy(self.ndi_find)


        #### NDI SEND ####
        self.send_settings = ndi.SendCreate()
        self.send_settings.ndi_name = 'dyna-projection'

        self.ndi_send = ndi.send_create(self.send_settings)
        self.video_frame = ndi.VideoFrameV2()

        self.ndi_recv_buffer = []
        self.nframe = np.zeros((self.res_y, self.res_x, 3),dtype=np.uint8)
        self.blank_frame = np.zeros((self.res_y, self.res_x, 3),dtype=np.uint8)

    def set_camera_res():
        pass
    @Slot()
    def stop(self):
        self.status = False
        
    @Slot()
    def run(self):
        self.status = True # thread state
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_y)

        isolated = self.blank_frame

        while self.status:
            success, frame = self.cap.read()
            t, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)

            if t == ndi.FRAME_TYPE_VIDEO:
                self.ndi_frame = np.copy(v.data)
                self.ndi_frame = cv2.resize(self.ndi_frame, (self.res_x, self.res_y))[:,:,:3]
                self.ndi_recv_buffer.append(self.ndi_frame)
                ndi.recv_free_video_v2(self.ndi_recv, v)
            if success:
                # start = time.perf_counter()
                results = self.model.predict(frame, verbose = False)
                # for result in results:
                result = results[0]

                if len(self.ndi_recv_buffer):
                    self.nframe = self.ndi_recv_buffer.pop() 

                if not self.camera_preview:
                    if result.masks is not None:
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
                        
                        mask = cv2.cvtColor(people_mask.cpu().numpy().astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        
                        if mask.size:
                            isolated = cv2.bitwise_and(mask, self.nframe)
                    elif counter>3: 
                        counter = 0
                        isolated = self.blank_frame
                    else: counter+=1
                else:
                    isolated = frame
                img = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
                self.video_frame.data = img
                self.video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

                ndi.send_send_video_v2(self.ndi_send, self.video_frame)          
            
            # Reading the image in RGB to display it
            if self.enable_preview:
                color_frame = cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB)

                h, w, ch = color_frame.shape
                img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)
                scaled_img = img.scaled(self.res_x, self.res_y, Qt.KeepAspectRatio)

                # Emit signal
                self.updateFrame.emit(scaled_img)
        # sys .exit(-1)

    @Slot()
    def enable_preview(self):
        self.camera_preview = True

    @Slot()
    def disable_preview(self):
        self.camera_preview = False
    
    @Slot()
    def set_ndi_source(self, ndi_source):
        ndi.recv_connect(self.ndi_recv, ndi_source)
        # print("wow this actually work")



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Dynamic Projection")
        self.setGeometry(0, 0, 800, 500)

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=qApp.quit)  # noqa: F821
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
                        triggered=qApp.aboutQt)  # noqa: F821
        self.menu_about.addAction(about)
        
        # self.menu_setting =

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Model group
        self.group_model = QGroupBox("NDI Source")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()

        if not ndi.initialize():
            sys.exit(1)

        #### NDI RECV ####
        self.ndi_find = ndi.find_create_v2()

        if self.ndi_find is None:
            sys.exit(1)

        self.sources = []

        self.combobox = QComboBox()
        # for source in self.sources:
        #     self.combobox.addItem(source.ndi_name)

        self.refresh_button = QPushButton("R")

        model_layout.addWidget(QLabel("Source:"), 10)
        model_layout.addWidget(self.combobox, 90)
        model_layout.addWidget(self.refresh_button)
        self.group_model.setLayout(model_layout)
        
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        # Checkbox
        self.checkbox = QCheckBox("Enable preview", self)
        buttons_layout.addWidget(self.checkbox)


        # Buttons group
        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)

        

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        
        
        # Connections
        self.checkbox.clicked.connect(self.checkbox_clicked)
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)
        self.refresh_button.clicked.connect(self.refresh_sources)
        self.combobox.currentIndexChanged.connect(self.set_ndi_source)

    @Slot()
    def checkbox_clicked(self, checked):
        if checked:
            self.th.enable_preview()
        else:
            self.th.disable_preview()
    
    @Slot()
    def set_ndi_source(self, source_id):
        self.th.set_ndi_source(self.sources[source_id])
    
    def refresh_sources(self):
        if not ndi.find_wait_for_sources(self.ndi_find, 1000):
            print("no change in source")
        else:
            self.sources = ndi.find_get_current_sources(self.ndi_find)
            self.combobox.clear()
            for source in self.sources:
                self.combobox.addItem(source.ndi_name)


    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.th.stop()
        time.sleep(1)
        self.th.quit()
        # Give time for the thread to finish
        time.sleep(1)

    @Slot()
    def start(self):
        print("Starting...")
        # if()
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        # self.th.set_file(self.combobox.currentText())
        # self.th.status = True
        self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    app.exec()