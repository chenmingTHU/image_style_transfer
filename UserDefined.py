from PyQt5.QtWidgets import QWidget, QPushButton, QProgressBar, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QMessageBox
import ImageViewer
from PyQt5.QtCore import QBasicTimer
from eval_arbitrary import *
import subprocess

class myUserDefined(QWidget):

    def __init__(self):
        super(QWidget, self).__init__()
        self.initUI()

    def initUI(self):
        content = 'pics/default.png'
        self.labelContent = ImageViewer.myImageViewer(content, 320, 240)
        style = 'pics/default_style.png'
        self.labelStyle = ImageViewer.myImageViewer(style, 320, 240)
        result = 'pics/default_null.png'
        self.labelResult = ImageViewer.myImageResult(result, 480, 360)
        self.defaultInpath = self.labelContent.getImagePath()
        self.defaultStylepath = self.labelStyle.getImagePath()

        self.qb1 = QPushButton('Transfer')
        self.qb1.setFixedSize(80, 20)
        self.qb1.clicked.connect(self.transfer)
        self.qb2 = QPushButton('Share')
        self.qb2.setFixedSize(80, 20)
        self.qb2.setDisabled(True)
        self.qb2.clicked.connect(self.share)
        self.timer = QBasicTimer()
        self.step = 0
        
        self.pb = QProgressBar()
        self.pb.setFixedSize(500, 10)

        self.hboxgroup1 = QGroupBox()
        self.hbox1 = QHBoxLayout()
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.labelContent)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.labelStyle)
        self.hbox1.addStretch(1)
        self.hboxgroup1.setLayout(self.hbox1)
        
        self.hboxgroup2 = QGroupBox()
        self.hbox2 = QHBoxLayout()
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.pb)
        self.hbox2.addStretch(1)
        self.hboxgroup2.setLayout(self.hbox2)

        self.hboxgroup3 = QGroupBox()
        self.hbox3 = QHBoxLayout()
        self.hbox3.addStretch(1)
        self.hbox3.addWidget(self.labelResult)
        self.hbox3.addStretch(1)
        self.hboxgroup3.setLayout(self.hbox3)

        self.hboxgroup4 = QGroupBox()
        self.hbox4 = QHBoxLayout()
        self.hbox4.addStretch(1)
        self.hbox4.addWidget(self.qb1)
        self.hbox4.addStretch(1)
        self.hbox4.addWidget(self.qb2)
        self.hbox4.addStretch(1)
        self.hboxgroup4.setLayout(self.hbox4)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.hboxgroup1)
        self.vbox.addWidget(self.hboxgroup2)
        self.vbox.addWidget(self.hboxgroup3)
        self.vbox.addWidget(self.hboxgroup4)
        self.setLayout(self.vbox)
        
    def setStepValue(self):
        #self.pb.setValue(self.step)
        pass

    def transfer(self):
        if self.labelContent.getImagePath() == self.defaultInpath:
            reply = QMessageBox.warning(self, "ERROR", "Please choose a input image!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if self.labelStyle.getImagePath() == self.defaultStylepath:
            reply = QMessageBox.warning(self, "ERROR", "Please choose a style image!", QMessageBox.Ok, QMessageBox.Ok)
            return
        if not self.timer.isActive():
            self.inpath = self.labelContent.getImagePath()
            self.outpath = "result/result.jpg"
            self.style = self.labelStyle.getImagePath()
            print(self.inpath)
            print(self.outpath)
            print(self.style)
            self.qb2.setDisabled(True)
            self.timer.start(100, self)
            self.ps = subprocess.Popen("python3 eval_arbitrary.py " + self.inpath + " " + self.style + " " + self.outpath, shell = True)
            self.qb2.setDisabled(True)
            self.timer.start(100, self)

    def timerEvent(self, a):
        if self.step >= 100:
            self.timer.stop()
            self.qb2.setDisabled(False)
            self.step = 0
            return
        self.step += 0.05
        if self.ps.poll() is not None:
            self.step = 100
            self.labelResult.changeImage(self.outpath)
        self.pb.setValue(self.step)

    def share(self):
        pass
