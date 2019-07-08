import sys
import os
import PIL
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
from PIL import Image
from PIL.ImageChops import multiply

class DrawingThread(QThread):

    def __init__(self, cropTool):
        QThread.__init__(self)
        self.cropTool = cropTool

    def __dle__(self):
        self.wait()

    def run(self):
        print("test run")
        while True:
            cursor = QCursor()
            pos = cursor.pos()
            print("test")
            self.cropTool.updateMirroredCursor(pos)
            self.sleep(0)


class CropTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.index = 0
        self.drawingWidth = 10
        self.imagePadding = 15
        self.imageWidth = 800
        self.imageHeight = 1000
        self.buttonHeight = 50
        self.buttonWidth = 200
        self.mainWidget = None
        self.img1 = None
        self.img2 = None
        self.opacitySlider = None
        self.mirroredCursorLabel = None
        self.image_names = []
        self.noCropImages = {}
        self.maskImages = {}
        self.newMaskImages = {}
        self.windowWidth = self.imageWidth * 2 + self.imagePadding * 3
        self.windowHeight = self.imageHeight + self.imagePadding * 3 + self.buttonHeight
        self.initUI()        
        self.drawThread = DrawingThread(self)
        self.drawThread.start()

    def mousePressEvent(self, QMouseEvent):
        print(QMouseEvent.pos())    
        overIm1 = self.cursorOverImage1(QMouseEvent.pos())
        overIm2 = self.cursorOverImage2(QMouseEvent.pos())
        print("Over 1: " + str(overIm1) + " Over 2: " + str(overIm2))
        if(overIm1):
            print("Im 1 relative pos: " + str(self.cursorPosOverImage1(QMouseEvent.pos())))
        if(overIm2):
            print("Im 2 relative pos: " + str(self.cursorPosOverImage2(QMouseEvent.pos())))

    def cursorOverImage1(self, point):
        if(point.x() > self.imagePadding and point.x() < self.imagePadding + self.imageWidth and point.y() > self.imagePadding and point.y() < self.imagePadding + self.imageHeight):
            return True
        return False
    def cursorOverImage2(self, point):  
        if(point.x() > self.imagePadding * 2 + self.imageWidth and point.x() < self.imagePadding * 2 + self.imageWidth * 2 and point.y() > self.imagePadding and point.y() < self.imagePadding + self.imageHeight):
            return True
        return False
    def cursorPosOverImage1(self, point):
        return QPoint(point.x()-self.imagePadding, point.y()-self.imagePadding)
    def cursorPosOverImage2(self, point):
        return QPoint(point.x()-(self.imagePadding * 2 + self.imageWidth), point.y()-self.imagePadding)
    def getCorrespondingPointOverImage1(self, point):
        relativePoint = self.cursorPosOverImage2(point)
        return QPoint(relativePoint.x() + self.imagePadding, relativePoint.y() + self.imagePadding)
    def getCorrespondingPointOverImage2(self, point):
        relativePoint = self.cursorPosOverImage1(point)
        return QPoint(relativePoint.x() + self.imagePadding * 2 + self.imageWidth, relativePoint.y() + self.imagePadding)
    
    def sliderValueChanged(self):
        self.updateMask()
        
    def initUI(self):        
        self.mainWidget = QWidget()
        self.mirroredCursorLabel = QLabel(self.mainWidget)
        self.mirroredCursorLabel.setPixmap(QPixmap("whiteRect.png"))
        self.img1 = QLabel(self.mainWidget)
        self.img2 = QLabel(self.mainWidget)

        self.img1.setGeometry(self.imagePadding, self.imagePadding, self.imageWidth, self.imageHeight)
        self.img2.setGeometry(self.imagePadding * 2 + self.imageWidth, self.imagePadding, self.imageWidth, self.imageHeight)
        #vbox = QVBoxLayout()
        #vbox.addWidget(l1)
        #vbox.addWidget(l2)
        self.opacitySlider = QSlider(Qt.Horizontal, self.mainWidget)
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(50)
        self.opacitySlider.setTickPosition(QSlider.TicksBelow)
        self.opacitySlider.setTickInterval(10)
        self.opacitySlider.setGeometry(self.windowWidth / 2 - self.buttonWidth / 2, self.windowHeight - self.imagePadding - self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.opacitySlider.valueChanged.connect(self.sliderValueChanged)
        #w.setLayout(vbox)
        self.setGeometry(0, 0, self.windowWidth, self.windowHeight)
        self.setWindowTitle("Interactive Cropping Tool")
        self.setCentralWidget(self.mainWidget)

    def setImageDirectories(self, noCropDir, maskDir):
        for image in os.listdir(noCropDir):
            im = PIL.Image.open(os.path.join(noCropDir, image))
            short_name = image.split(".")[0]
            self.noCropImages[short_name] = im
            self.image_names.append(short_name)
        for image in os.listdir(maskDir):
            im = PIL.Image.open(os.path.join(maskDir, image))
            short_name = image.split(".")[0]
            self.maskImages[short_name] = im
            self.newMaskImages[short_name] = im
    
    def setPics(self, index):
        self.index = index
        print("Setting pic to " + str(self.image_names[self.index]))
        im = self.newMaskImages[self.image_names[self.index]]
        im = np.array(im)
        im = np.minimum(np.ones_like(im) * 255, im + (self.opacitySlider.value() / 100) * 255)
        im = PIL.Image.fromarray(np.uint8(im))        
        qim1 = ImageQt(multiply(self.noCropImages[self.image_names[self.index]], im))
        qim2 = ImageQt(self.maskImages[self.image_names[self.index]])
        pic1 = QPixmap.fromImage(qim1)
        pic2 = QPixmap.fromImage(qim2)
        self.img1.setPixmap(pic1)
        self.img2.setPixmap(pic2)

    def updateMask(self):
        im = self.newMaskImages[self.image_names[self.index]]
        im = np.array(im)
        im = np.minimum(np.ones_like(im) * 255, im + (self.opacitySlider.value() / 100) * 255)
        im = PIL.Image.fromarray(np.uint8(im))        
        qim1 = ImageQt(multiply(self.noCropImages[self.image_names[self.index]], im))
        pic1 = QPixmap.fromImage(qim1)
        self.img1.setPixmap(pic1)

    def updateMirroredCursor(self, point):
        if(self.cursorOverImage1(point)):
            mirrorPoint = self.getCorrespondingPointOverImage2(point)
            self.mirroredCursorLabel.setVisible(True)
            self.mirroredCursorLabel.setGeometry(mirrorPoint.x() - self.drawingWidth / 2, mirrorPoint.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
        elif(self.cursorOverImage2(point)):
            mirrorPoint = self.getCorrespondingPointOverImage1(point)
            self.mirroredCursorLabel.setVisible(True)
            self.mirroredCursorLabel.setGeometry(mirrorPoint.x() - self.drawingWidth / 2, mirrorPoint.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
        else:
            self.mirroredCursorLabel.setVisible(False)
def main():
    app = QApplication(sys.argv)
    win = CropTool()
    win.setImageDirectories("../Images/NoCropForAnalysis/Normal", "../Images/AutoCropMasks/Normal")
    win.setPics(0)
    win.show()
    app.exit(app.exec_())
    
if __name__ == '__main__':
    main()