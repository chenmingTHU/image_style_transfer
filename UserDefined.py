from PyQt5.QtWidgets import QWidget, QPushButton, QProgressBar, QLabel, QHBoxLayout, QVBoxLayout, QGroupBox, QMessageBox, QSpinBox, QSlider, QInputDialog
import ImageViewer
from PyQt5.Qt import Qt
from PyQt5.QtCore import QBasicTimer
from eval_arbitrary import *
import subprocess
import os
from weibo import *
import requests
from requests_toolbelt import MultipartEncoder

class myUserDefined(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__()
        self.parent = parent
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

        self.sp = QSpinBox(self)
        #self.sp.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.sl = QSlider(Qt.Vertical, self)
        self.sp.setRange(0, 100)
        self.sp.setSingleStep(1)
        self.sl.setFixedSize(10, 200)
        self.sp.setWrapping(True)
        self.sp.setValue(0)
        self.sp.setSuffix(" %")
        self.sl.setRange(0, 100)
        self.sl.setValue(0)
        self.sp.valueChanged.connect(self.slider_changevalue)
        self.sl.valueChanged.connect(self.spinbox_changevalue)
        self.hboxscrollgroup = QGroupBox()
        self.hboxscroll = QHBoxLayout()
        self.hboxscroll.addStretch(1)
        self.hboxscroll.addWidget(self.sl)
        self.hboxscroll.addStretch(1)
        self.hboxscrollgroup.setLayout(self.hboxscroll)
        self.vboxscrollgroup = QGroupBox()
        self.vboxscroll = QVBoxLayout()
        self.vboxscroll.addStretch(1)
        self.vboxscroll.addWidget(self.sp)
        self.vboxscroll.addStretch(1)
        self.vboxscroll.addWidget(self.hboxscrollgroup)
        self.vboxscroll.addStretch(1)
        self.vboxscrollgroup.setLayout(self.vboxscroll)

        self.hboxgroup1 = QGroupBox()
        self.hbox1 = QHBoxLayout()
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.labelContent)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.vboxscrollgroup)
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
            self.outpath = "result/" + self.labelContent.getImageName() + "_transfered_arbitrary.jpg"
            self.outname = self.labelContent.getImageName() + "_transfered_arbitrary"
            while os.path.exists("result/" + self.outname + ".jpg"):
                self.outname = self.outname + "_1"
            self.outpath = "result/" + self.outname + ".jpg"
            self.style = self.labelStyle.getImagePath()
            self.width = self.labelContent.getImageWidth()
            self.height = self.labelContent.getImageHeight()
            self.ratio = float(self.width) / float(self.height)
            self.maxheight = 600
            self.maxwidth = 800
            if self.height > self.maxheight:
                self.height = self.maxheight
                self.width = int(self.height * self.ratio)
            if self.width > self.maxwidth:
                self.width = self.maxwidth
                self.height = int(self.width / self.ratio)
            self.qb2.setDisabled(True)
            self.timer.start(100, self)
            self.ps = subprocess.Popen("python3 eval_arb.py " + self.inpath + " " + self.style + " " + self.outpath + " " + str(self.height) + " " + str(self.width), shell = True)
            self.qb2.setDisabled(True)
            self.timer.start(100, self)

    def timerEvent(self, a):
        if self.step >= 100:
            self.timer.stop()
            self.qb2.setDisabled(False)
            self.step = 0
            return
        self.step += 0.35
        if self.ps.poll() is not None:
            self.step = 100
            self.labelResult.changeImage(self.outpath)
            self.parent.newHistory()
        self.pb.setValue(self.step)

    def share(self):
        sender = self.sender()
        if sender == self.qb2:
            text, ok = QInputDialog.getText(self, '分享到微博', '请输入您想说的话：')
            if ok:
                post_a_pic(self.outpath, text)

    def spinbox_changevalue(self,value):
        self.sp.setValue(value)

    def slider_changevalue(self,value):
        self.sl.setValue(value)
