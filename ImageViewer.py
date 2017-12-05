from PyQt5.QtWidgets import QLabel, QFileDialog, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt

class newWidget(QWidget):

    def __init__(self, image):
        super(QWidget, self).__init__()
        self.setFixedSize(image.width(), image.height())
        self.label = QLabel(self)
        self.label.setFixedSize(image.width(), image.height())
        self.label.setPixmap(QPixmap.fromImage(image))

class myImageViewer(QLabel):

    def __init__(self, image, width, height):
        super(QLabel, self).__init__()
        self.imagePath = image
        self.image = QImage(image)
        self.width = width
        self.height = height
        self.setFixedSize(self.width, self.height)
        self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image)) 
    
    def mousePressEvent(self, e):
        newImage = QFileDialog.getOpenFileName(self, 'open file', './', 'Images (*.png *.jpg)')[0]
        if len(newImage):
            self.imagePath = newImage
            self.image = QImage(newImage)
            self.image = self.image.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(QPixmap.fromImage(self.image))

    def getImagePath(self):
        return self.imagePath

class myImageResult(QLabel):

    def __init__(self, image, width, height):
        super(QLabel, self).__init__()
        self.imagePath = image
        self.origImage = QImage(image)
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
        self.origImage = QImage(image)
        self.setFixedSize(self.width, self.height)
        self.image = self.origImage.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.newWidget = newWidget(self.origImage)
