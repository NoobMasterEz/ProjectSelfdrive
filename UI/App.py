import cv2
import sys
import UI.ThreadVideo as ThreadVideo
import UI.ThreadSteeringWheel as ThreadSteeringWheel
from PyQt5.QtWidgets import  *
from PyQt5.QtCore import  pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

class App(QWidget):

    """
    Doc : 
    """
    def __init__(self):
        super().__init__()
        
        self.display_width=1080
        self.display_height=720

        
        self.layout = QGridLayout()
        self.setLayout(self.layout)


        #set title
        self.setWindowTitle("OpenCv")
        self.setGeometry(100,100,self.display_width,self.display_height)

        self.groupbox = QGroupBox("System Automation")

        #Create label 
        self.label= QLabel(self)

        self.label1=QLabel(self)
        
        self.layout.addWidget(self.groupbox)
        lay = QVBoxLayout()
        self.groupbox.setLayout(lay)

        lay.addWidget(self.label)
        lay.addWidget(self.label1)
        
        

        self.th = ThreadSteeringWheel.Thread()
        self.th1 = ThreadVideo.Thread1()
        
        self.th.changePixmap.connect(self.setImage)
        self.th1.changePixmap1.connect(self.setImage1)
        self.th.start()
        self.th1.start()
        
        self.show()


    @pyqtSlot(QImage)
    def setImage(self,image):
        
        self.label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setImage1(self,image):
        self.label1.setPixmap(QPixmap.fromImage(image))
        
   

def main():
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

