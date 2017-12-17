from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QPushButton, QProgressBar, QMessageBox
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPixmap, QImage
from ImageViewer import *
import os
import glob

class myHistory(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.imageList = []
        self.imageList = glob.glob("result/*.jpg")
        self.imageList.sort(key = os.path.getmtime)
        self.imageList.reverse()
        num = len(self.imageList)
        self.labelList = []
        self.hboxgroupList = []
        self.hboxList = []
        self.vboxgroup = QGroupBox()
        self.vbox = QVBoxLayout()
        self.vboxLayout = QVBoxLayout()
        for i in range(num):
            self.labelList.append(myImageResult(self.imageList[i], 320, 240))
        if num % 2 == 1:
            self.labelList.append(myImageBlank("pics/null.png", 320, 240))
            num += 1
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

    def renew(self):
        self.setUpdatesEnabled(False)
        self.initUI()
        self.setUpdatesEnabled(True)
