from PyQt5.QtWidgets import QLabel, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt
from NewWindow import *

def getName(path):
    name = ''
    path = path.split('/')[-1]
    path = path.split('.')[0:-1]
    for i in path:
        name += i
    return name

class myImageViewer(QLabel):

    def __init__(self, image, width, height):
        super(QLabel, self).__init__()
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.image = QImage(image)
        self.origWidth = self.image.width()
        self.origHeight = self.image.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image)) 
    
    def mousePressEvent(self, e):
        newImage = QFileDialog.getOpenFileName(self, 'open file', './', 'Images (*.png *.jpg)')[0]
        if len(newImage):
            self.imagePath = newImage
            self.name = getName(self.imagePath)
            self.image = QImage(newImage)
            self.origWidth = self.image.width()
            self.origHeight = self.image.height()
            self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(self.image))

    def getImagePath(self):
        return self.imagePath

    def getImageName(self):
        return self.name
    
    def getImageWidth(self):
        return self.origWidth
    
    def getImageHeight(self):
        return self.origHeight

class myImageResult(QLabel):

    def __init__(self, image, width, height):
        super(QLabel, self).__init__()
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.origImage = QImage(image)
        self.origWidth = self.origImage.width()
        self.origHeight = self.origImage.height()
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.newWidget = newWidget(self.origImage)
    
    def mousePressEvent(self, e):
        self.newWidget.show()
    
    def changeImage(self, image):
        self.imagePath = image
        self.name = getName(self.imagePath)
        self.origImage = QImage(image)
        self.origWidth = self.origImage.width()
        self.origHeight = self.origImage.height()
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.newWidget = newWidget(self.origImage)
