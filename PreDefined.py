from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QScrollArea, QPushButton, QProgressBar, QMessageBox
from PyQt5.Qt import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QBasicTimer
from ImageViewer import *
from eval_pretrained import *
import subprocess

global flag 
flag = 0
class picLabel(QLabel):

    def __init__(self, parent, index, picName = "default.png", width = 200, height = 200):
        super(QLabel, self).__init__()
        self.clicked = False
        self.parent = parent
        self.index = index
        self.picName = picName
        self.width = width
        self.height = height
        self.initUI()

    def initUI(self):
        self.pic = QImage(self.picName)
        self.setFixedSize(self.width, self.height)
        self.pic = self.pic.scaled(self.width, self.height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(QPixmap.fromImage(self.pic))
        self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)") 

    def mousePressEvent(self, e):
        if self.index == 0: return
        global flag
        if flag == 0:
            flag = self.index
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 100%)") 
        elif flag == self.index:
            flag = 0
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)") 
        else:
            self.parent.VBoxGroups[flag - 1].prevPic.setStyleSheet("border: 3px solid rgba(0, 0, 255, 0%)")
            self.setStyleSheet("border: 3px solid rgba(0, 0, 255, 100%)") 
            flag = self.index

class preStyle(QGroupBox):

    def __init__(self, parent, index, labelText = "youhua", picName = "default.png", width = 200, height = 200):
        super(QGroupBox, self).__init__()
        self.prevPic = picLabel(parent, index, picName, width, height)
        self.labelText = labelText
        self.initUI()

    def initUI(self):
        self.label = QLabel()
        self.label.resize(20, 5)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText(self.labelText)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.label)
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.prevPic)
        self.setLayout(self.vbox)

class myPreDefined(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.style = ["style1", "style2", "style3", "style4", "style5", "style6", "style7", "style8", "style9"]
        self.picPath = ["pics/style1.jpg", "pics/style2.jpg", "pics/style3.jpg", "pics/style4.jpg", "pics/style5.jpg", "pics/style6.jpg", "pics/style7.jpg", "pics/style8.jpg", "pics/style9.jpg"]
        self.stylePath = ["pretrained_model/style1", "pretrained_model/style2", "pretrained_model/style3", "pretrained_model/style4", "pretrained_model/style5", "pretrained_model/style6", "pretrained_model/style7", "pretrained_model/style8", "pretrained_model/style9"]
        self.nStyle = len(self.style)
        self.initUI()

    def initUI(self):
        self.inputPic = myImageViewer("pics/default.png", 400, 300)
        self.outputPic = myImageResult("pics/default_null.png", 400, 300)
        self.defaultInpath = self.inputPic.getImagePath()

        self.VBoxGroups = []
        for i in range(self.nStyle):
            self.VBoxGroups.append(preStyle(self, i + 1, self.style[i], self.picPath[i], 200, 200))

        self.HBoxGroupScroll = QGroupBox()
        self.HBox1 = QHBoxLayout()
        self.HBox1.addStretch(1)
        for group in self.VBoxGroups:
            self.HBox1.addWidget(group)
            self.HBox1.addStretch(1)
        self.HBoxGroupScroll.setLayout(self.HBox1)

        self.transButton = QPushButton('Transfer')
        self.transButton.setFixedSize(80, 20)
        self.transButton.clicked.connect(self.transfer)
        self.shareButton = QPushButton('Share')
        self.shareButton.setFixedSize(80, 20)
        self.shareButton.clicked.connect(self.share)
        self.shareButton.setDisabled(True)
        self.timer = QBasicTimer()
        self.step = 0

        self.HBoxGroupButton = QGroupBox()
        self.HBoxButton = QHBoxLayout()
        self.HBoxButton.addStretch(1)
        self.HBoxButton.addWidget(self.transButton)
        self.HBoxButton.addStretch(1)
        self.HBoxButton.addWidget(self.shareButton)
        self.HBoxButton.addStretch(1)
        self.HBoxGroupButton.setLayout(self.HBoxButton)
        
        self.progBar = QProgressBar()
        self.progBar.setFixedSize(500, 10)
        self.HBoxGroupBar = QGroupBox()
        self.HBoxBar = QHBoxLayout()
        self.HBoxBar.addStretch(1)
        self.HBoxBar.addWidget(self.progBar)
        self.HBoxBar.addStretch(1)
        self.HBoxGroupBar.setLayout(self.HBoxBar)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.HBoxGroupScroll) 
        self.scroll.setAutoFillBackground(True)
        self.scroll.setWidgetResizable(True)
        #self.HBoxGroupUp = QGroupBox()
        #self.HBoxUp = QHBoxLayout()
        #self.HBoxUp.addWidget(self.scroll)
        #self.HBoxGroupUp.setLayout(self.HBoxUp)

        self.HBoxGroupDown = QGroupBox()
        self.HBox2 = QHBoxLayout()
        self.HBox2.addStretch(1)
        self.HBox2.addWidget(self.inputPic)
        self.HBox2.addStretch(1)
        self.HBox2.addWidget(self.outputPic)
        self.HBox2.addStretch(1)
        self.HBoxGroupDown.setLayout(self.HBox2)

        self.VBox = QVBoxLayout()
        self.VBox.addStretch(1)
        self.VBox.addWidget(self.scroll)
        self.VBox.addStretch(1)
        self.VBox.addWidget(self.HBoxGroupBar)
        self.VBox.addStretch(1)
        self.VBox.addWidget(self.HBoxGroupDown)
        self.VBox.addStretch(1)
        self.VBox.addWidget(self.HBoxGroupButton)
        self.VBox.addStretch(1)
        self.setLayout(self.VBox)

    def transfer(self):
        global flag
        if flag == 0:
            reply = QMessageBox.warning(self, "ERROR", "Please choose a style!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.inputPic.getImagePath() == self.defaultInpath:
            reply = QMessageBox.warning(self, "ERROR", "Please choose a input image!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not self.timer.isActive():
            self.inpath = self.inputPic.getImagePath()
            self.outpath = "result/result.jpg"
            self.style = self.stylePath[flag - 1]
            self.shareButton.setDisabled(True)
            self.timer.start(100, self)
            self.ps = subprocess.Popen("python3 eval_pretrained.py " + self.inpath + " " + self.outpath + " " + self.style, shell = True)

    def timerEvent(self, a):
        if self.step >= 100:
            self.timer.stop()
            self.shareButton.setDisabled(False)
            self.step = 0
            return
        self.step += 0.2
        if self.ps.poll() is not None:
            self.step = 100
            self.outputPic.changeImage(self.outpath)
        self.progBar.setValue(self.step)

    def share(self):
        pass
