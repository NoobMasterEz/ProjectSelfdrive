import cv2
import socket
import os 
from subprocess import call
import os
from PyQt5.QtCore import QThread,Qt,pyqtSignal
from PyQt5.QtGui import QImage
import json 

f=open("Config/config.json")
data=json.load(f)

#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True



localIP     = data["SteeringWheel"]["localIP"]

localPort   = data["SteeringWheel"]["localPort"]

bufferSize  = data["SteeringWheel"]["bufferSize"]


 



 

# Create a datagram socket

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

 

# Bind to address and ip

UDPServerSocket.bind((localIP, localPort))



class Thread(QThread):
    
    """
    docstring
    """
    changePixmap = pyqtSignal(QImage)

    
    def __init__(self,parent=None):
        QThread.__init__(self,parent)
        self.smoothed_angle = 0


    def run(self):
        img = cv2.imread(data["SteeringWheel"]["img"]
)
        print(img.shape)
        
        rows,cols,ch = img.shape
        # SOCK_DGRAM is the socket type to use for UDP sockets
        
        while(True):
            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            degrees = float(bytesAddressPair[0])
            #print(degrees)
            if not windows:
                call("clear")
            print("Predicted steering angle: " + str(degrees) + " degrees")
            
            #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
            #and the predicted angle
            self.smoothed_angle += 0.2 * pow(abs((degrees - self.smoothed_angle)), 2.0 / 3.0) * (degrees - self.smoothed_angle) / abs(degrees - self.smoothed_angle)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),-self.smoothed_angle,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            #print(dst.shape)
            #ret,rgbimage = cv2.threshold(rgbimage,127,255,cv2.THRESH_BINARY)
            h, w,ch = dst.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(dst.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(400, 400, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)