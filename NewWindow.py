from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QSlider
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication, QPainter, QPen
from PyQt5.QtCore import QRect
from PyQt5.Qt import Qt
from ImageProcessing import *
import shutil
import math

class newLabel(QLabel):

    def __init__(self, image):
        super(QLabel, self).__init__()
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.flag = False
        self.imagePath = image
        self.newImagePath = "temp/newImage.jpg"
        shutil.copy(self.imagePath, self.newImagePath)
        self.image = QImage(image)
        self.width = self.image.width()
        self.height = self.image.height()
        self.setFixedSize(self.width, self.height)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)

    def changeImage(self, image):
        self.imagePath = image
        self.image = QImage(image)
        self.width = self.image.width()
        self.height = self.image.height()
        self.setFixedSize(self.width, self.height)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setAlignment(Qt.AlignCenter)

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getImagePath(self):
        return self.imagePath

    def mousePressEvent(self,event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    def mouseReleaseEvent(self,event):
        self.flag = False

    def mouseMoveEvent(self,event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        rect =QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        painter.drawRect(rect)
        #pqscreen  = QGuiApplication.primaryScreen()
        #pixmap2 = pqscreen.grabWindow(self.winId(), self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        #pixmap2.save('0000.png')

class newWidget(QWidget):

    def __init__(self, image):
        super(QWidget, self).__init__()
        self.label = newLabel(image)
        self.setFixedSize(self.label.getWidth()+160, self.label.getHeight()+100)

        self.updownButton = QPushButton('上下翻转')
        self.updownButton.setFixedSize(80, 30)
        self.updownButton.clicked.connect(self.updown)
        self.leftrightButton = QPushButton('左右翻转')
        self.leftrightButton.setFixedSize(80, 30)
        self.leftrightButton.clicked.connect(self.leftright)
        self.clockButton = QPushButton('顺时针旋转')
        self.clockButton.setFixedSize(80, 30)
        self.clockButton.clicked.connect(self.clock)
        self.anticlockButton = QPushButton('逆时针旋转')
        self.anticlockButton.setFixedSize(80, 30)
        self.anticlockButton.clicked.connect(self.anticlock)
        self.brightLabel = QLabel()
        self.brightLabel.resize(80, 20)
        self.brightLabel.setText("亮度")
        self.brightLabel.setAlignment(Qt.AlignLeft)
        self.brightSlider = QSlider(Qt.Horizontal)
        self.brightSlider.setRange(0, 100)
        self.brightSlider.setValue(50)
        self.brightSlider.valueChanged.connect(self.enhance)
        self.sharpLabel = QLabel()
        self.sharpLabel.resize(80, 20)
        self.sharpLabel.setText("锐利度")
        self.sharpLabel.setAlignment(Qt.AlignLeft)
        self.sharpSlider = QSlider(Qt.Horizontal)
        self.sharpSlider.setRange(0, 100)
        self.sharpSlider.setValue(50)
        self.sharpSlider.valueChanged.connect(self.enhance)
        self.contrastLabel = QLabel()
        self.contrastLabel.resize(80, 20)
        self.contrastLabel.setText("对比度")
        self.contrastLabel.setAlignment(Qt.AlignLeft)
        self.contrastSlider = QSlider(Qt.Horizontal)
        self.contrastSlider.setRange(0, 100)
        self.contrastSlider.setValue(50)
        self.contrastSlider.valueChanged.connect(self.enhance)

        self.vboxgroup = QGroupBox()
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.updownButton)
        self.vbox.addWidget(self.leftrightButton)
        self.vbox.addWidget(self.clockButton)
        self.vbox.addWidget(self.anticlockButton)
        self.vbox.addWidget(self.brightLabel)
        self.vbox.addWidget(self.brightSlider)
        self.vbox.addWidget(self.sharpLabel)
        self.vbox.addWidget(self.sharpSlider)
        self.vbox.addWidget(self.contrastLabel)
        self.vbox.addWidget(self.contrastSlider)
        self.vbox.addStretch(1)
        self.vboxgroup.setLayout(self.vbox)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.label)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.vboxgroup)
        self.hbox.addStretch(1)
        self.setLayout(self.hbox)

    def change(self, image):
        self.label.changeImage(image)
        self.setFixedSize(self.label.getWidth()+160, self.label.getHeight()+100)

    def updown(self):
        img_flip_up(self.label.getImagePath())
        self.change('temp/temp.jpg')
        img_flip_up(self.label.newImagePath, self.label.newImagePath)
    
    def leftright(self):
        img_flip_left(self.label.getImagePath())
        self.change('temp/temp.jpg')
        img_flip_left(self.label.newImagePath, self.label.newImagePath)

    def clock(self):
        img_rotate_cw(self.label.getImagePath())
        self.change('temp/temp.jpg')
        img_rotate_cw(self.label.newImagePath, self.label.newImagePath)

    def anticlock(self):
        img_rotate_ccw(self.label.getImagePath())
        self.change('temp/temp.jpg')
        img_rotate_ccw(self.label.newImagePath, self.label.newImagePath)

    def enhance(self):
        bright = math.exp((self.brightSlider.value()-50)/50.)
        sharp = math.exp((self.sharpSlider.value()-50)/25.)
        contrast = math.exp((self.contrastSlider.value()-50)/50.)
        img_enhance(self.label.newImagePath, bright, sharp, contrast)
        self.change('temp/temp.jpg')
