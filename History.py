from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QPushButton, QProgressBar, QMessageBox
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPixmap, QImage
from ImageViewer import *

class myHistory(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.initUI()

    def initUI(self):
        num = 8
        image = "pics/default_null.png"
        imageList = []
        imageList.append(image)
        self.imageList = imageList * num
        self.labelList = []
        self.hboxgroupList = []
        self.hboxList = []
        self.vboxgroup = QGroupBox()
        self.vbox = QVBoxLayout()
        self.vboxLayout = QVBoxLayout()
        for i in range(num):
            self.labelList.append(myImageResult(self.imageList[i], 320, 240))
        for i in range(int((num+1)/2)):
            self.hboxgroupList.append(QGroupBox())
            self.hboxList.append(QHBoxLayout())
            self.hboxList[i].addStretch(1)
            self.hboxList[i].addWidget(self.labelList[i*2])
            self.hboxList[i].addStretch(1)
            self.hboxList[i].addWidget(self.labelList[i*2+1])
            self.hboxList[i].addStretch(1)
            self.hboxgroupList[i].setLayout(self.hboxList[i])
        for i in range(int((num+1)/2)):
            self.vbox.addWidget(self.hboxgroupList[i])
        self.vboxgroup.setLayout(self.vbox)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.vboxgroup)
        self.scroll.setAutoFillBackground(True)
        self.scroll.setWidgetResizable(True)
        self.vboxLayout.addWidget(self.scroll)
        self.setLayout(self.vboxLayout)
