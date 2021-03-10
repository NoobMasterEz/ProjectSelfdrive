import cv2
from subprocess import call
import os
from PyQt5.QtCore import QThread,Qt,pyqtSignal
from PyQt5.QtGui import QImage
import pathlib
from urllib.request import urlopen
import numpy as np 
import os
import json 

class Thread1(QThread):
    
    """
    docstring
    """
    changePixmap1 = pyqtSignal(QImage)

    def __init__(self,parent=None):
        QThread.__init__(self,parent)


    def run(self):

        f=open("Config/config.json")
        data=json.load(f)
        stream = cv2.VideoCapture(data["Video"]["ip_cam"])
        while(True):
                ret, frame =stream.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w,ch = frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
                self.changePixmap1.emit(p)