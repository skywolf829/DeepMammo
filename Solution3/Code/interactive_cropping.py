import sys
import os
import PIL
import atexit
import numpy as np
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL.ImageQt import ImageQt
from PIL import Image
from PIL.ImageChops import multiply
from threading import Thread
from time import sleep

class CropTool(QMainWindow):

    def __init__(self):
        super().__init__()
        self.running = True
        self.index = 0
        self.drawingWidth = 100
        self.imagePadding = 15
        self.imageWidth = 800
        self.imageHeight = 1000
        self.buttonHeight = 50
        self.buttonWidth = 200
        self.drawing = True
        self.erasing = False
        self.mousePressed = False
        self.mainWidget = None
        self.img1 = None
        self.img2 = None
        self.opacitySlider = None
        self.cursorLabel= None
        self.mirroredCursorLabel = None
        self.image_names = []
        self.noCropImages = {}
        self.maskImages = {}
        self.newMaskImages = {}
        self.windowWidth = self.imageWidth * 2 + self.imagePadding * 3
        self.windowHeight = self.imageHeight + self.imagePadding * 3 + self.buttonHeight
        self.initUI()        
        

    def mousePressEvent(self, QMouseEvent):
        print("Mouse Pressed " + str(QMouseEvent.pos()))
        self.mousePressed = True

    def mouseReleaseEvent(self, QMouseEvent):
        print("Mouse Released " + str(QMouseEvent.pos()))
        self.mousePressed = False

    def cursorOverImage1(self, point):
        return point.x() > self.imagePadding and point.x() < self.imagePadding + self.imageWidth and point.y() > self.imagePadding and point.y() < self.imagePadding + self.imageHeight
    def cursorOverImage2(self, point):  
        return point.x() > self.imagePadding * 2 + self.imageWidth and point.x() < self.imagePadding * 2 + self.imageWidth * 2 and point.y() > self.imagePadding and point.y() < self.imagePadding + self.imageHeight
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
        
        self.drawingButton = QPushButton('Draw', self.mainWidget)
        self.drawingButton.setGeometry(self.windowWidth / 2 + self.buttonWidth + self.imagePadding, self.windowHeight - self.imagePadding - self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.drawingButton.clicked.connect(self.drawButtonClicked)
        
        self.erasingButton = QPushButton('Erase', self.mainWidget)
        self.erasingButton.setGeometry(self.windowWidth / 2 + self.buttonWidth * 2 + self.imagePadding * 2, self.windowHeight - self.imagePadding - self.buttonHeight, self.buttonWidth, self.buttonHeight)
        self.erasingButton.clicked.connect(self.eraseButtonClicked)

        self.mirroredCursorLabel = QLabel(self.mainWidget)
        self.mirroredCursorLabel.setPixmap(QPixmap("whiteRect.png").scaled(self.drawingWidth, self.drawingWidth))

        self.cursorLabel = QLabel(self.mainWidget)
        self.cursorLabel.setPixmap(QPixmap("whiteRect.png").scaled(self.drawingWidth, self.drawingWidth))
        #w.setLayout(vbox)
        self.setGeometry(0, 0, self.windowWidth, self.windowHeight)
        self.setWindowTitle("Interactive Cropping Tool")
        self.setCentralWidget(self.mainWidget)

    def drawButtonClicked(self):
        self.drawing = True
        self.erasing = False
        self.mirroredCursorLabel.setPixmap(QPixmap("whiteRect.png").scaled(self.drawingWidth, self.drawingWidth))
        self.cursorLabel.setPixmap(QPixmap("whiteRect.png").scaled(self.drawingWidth, self.drawingWidth))

    def eraseButtonClicked(self):
        self.drawing = False
        self.erasing = True
        self.mirroredCursorLabel.setPixmap(QPixmap("blackRect.png").scaled(self.drawingWidth, self.drawingWidth))
        self.cursorLabel.setPixmap(QPixmap("blackRect.png").scaled(self.drawingWidth, self.drawingWidth))

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
        point = self.mainWidget.mapFromGlobal(point)
        
        if(self.cursorOverImage1(point)):
            print("////////////over image 1 - " + str(int(round(time.time() * 1000))))
            mirrorPoint = self.getCorrespondingPointOverImage2(point)
            print("Got mirror point")
            self.mirroredCursorLabel.setVisible(True)
            self.mirroredCursorLabel.setGeometry(mirrorPoint.x() - self.drawingWidth / 2, mirrorPoint.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
            print("Finished mirrored cursor")
            self.cursorLabel.setVisible(True)
            self.cursorLabel.setGeometry(point.x() - self.drawingWidth / 2, point.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
            print("///////////finished draw - " + str(int(round(time.time() * 1000))))
        """
        elif(self.cursorOverImage2(point)):
            print("over image 2")
            mirrorPoint = self.getCorrespondingPointOverImage1(point)
            self.mirroredCursorLabel.setVisible(True)
            self.mirroredCursorLabel.setGeometry(mirrorPoint.x() - self.drawingWidth / 2, mirrorPoint.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
            self.cursorLabel.setVisible(True)
            self.cursorLabel.setGeometry(point.x() - self.drawingWidth / 2, point.y() - self.drawingWidth / 2, self.drawingWidth, self.drawingWidth)
        else:
            self.mirroredCursorLabel.setVisible(False)
            self.cursorLabel.setVisible(False)
        """

    def updateDrawing(self, point):
        point = self.mainWidget.mapFromGlobal(point)
        if self.mousePressed:
            if self.cursorOverImage1(point):
                relativePosOverImage2 = self.cursorPosOverImage1(point)
                if self.drawing:
                    self.drawOnMask(relativePosOverImage2)
                elif self.erasing:
                    self.eraseOnMask(relativePosOverImage2)
            elif self.cursorOverImage2(point):
                relativePosOverImage2 = self.cursorPosOverImage2(point)
                if self.drawing:
                    self.drawOnMask(relativePosOverImage2)
                elif self.erasing:
                    self.eraseOnMask(relativePosOverImage2)

    def drawOnMask(self, point):
        im = self.newMaskImages[self.image_names[self.index]]
        im = np.array(im)  
        im[max(0, int(point.y() - self.drawingWidth / 2)):min(self.imageHeight, int(point.y() + self.drawingWidth / 2)),max(0, int(point.x() - self.drawingWidth / 2)):min(self.imageWidth, int(point.x() + self.drawingWidth / 2))] = 255       
        im = PIL.Image.fromarray(np.uint8(im))    
        self.newMaskImages[self.image_names[self.index]] = im
        #self.updateMask()        
        #self.img2.setPixmap(QPixmap.fromImage(ImageQt(im)))

    def eraseOnMask(self, point):
        im = self.newMaskImages[self.image_names[self.index]]
        im = np.array(im)  
        #print(point)
        im[max(0, int(point.y() - self.drawingWidth / 2)):min(self.imageHeight, int(point.y() + self.drawingWidth / 2)),max(0, int(point.x() - self.drawingWidth / 2)):min(self.imageWidth, int(point.x() + self.drawingWidth / 2))] = 0     
        im = PIL.Image.fromarray(np.uint8(im))    
        self.newMaskImages[self.image_names[self.index]] = im
        #self.updateMask()        
        #self.img2.setPixmap(QPixmap.fromImage(ImageQt(im)))
    def update(self):
        cursor = QCursor()
        while self.running:
            pos = cursor.pos()
            self.updateDrawing(pos)
            self.updateMirroredCursor(pos)
            sleep(0.01)
            
    def closeEvent(self, event):
        self.running = False
        event.accept()

def main():
    app = QApplication(sys.argv)
    win = CropTool()
    win.setImageDirectories("../Images/NoCropForAnalysis/Normal", "../Images/AutoCropMasks/Normal")
    win.setPics(0)
    win.show()
    thread = Thread(target = win.update)
    thread.start()
    app.exit(app.exec_())
    
if __name__ == '__main__':
    main()