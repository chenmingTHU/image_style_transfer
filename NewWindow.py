from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QPushButton, QSlider, QComboBox, QFileDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication, QPainter, QPen
from PyQt5.QtCore import QRect
from PyQt5.Qt import Qt
from ImageProcessing import *
import shutil
import math
from weibo import *
import requests
from requests_toolbelt import MultipartEncoder

class newLabel(QLabel):

    def __init__(self, image):
        super(QLabel, self).__init__()
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.flag = False
        self.switch = False
        self.ok = False
        self.origPath = image
        self.imagePath = image
        self.newImagePath = "temp/newImage.jpg"
        shutil.copy(self.imagePath, self.newImagePath)
        self.image = QImage(image)
        self.width = self.image.width()
        self.height = self.image.height()
        self.origWidth = self.width
        self.origHeight = self.height
        self.setFixedSize(self.width, self.height)
        #self.image = self.image.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pixmap = QPixmap.fromImage(self.image)
        self.setPixmap(self.pixmap)
        self.setAlignment(Qt.AlignCenter)

    def changeImage(self, image):
        self.imagePath = image
        self.image = QImage(image)
        self.width = self.image.width()
        self.height = self.image.height()
        self.setFixedSize(self.width, self.height)
        #self.image = self.image.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pixmap = QPixmap.fromImage(self.image)
        self.setPixmap(self.pixmap)
        self.setAlignment(Qt.AlignCenter)

    def resetCopy(self):
        shutil.copy(self.origPath, self.newImagePath)

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getImagePath(self):
        return self.imagePath

    def getOrigWidth(self):
        return self.origWidth

    def getOrigHeight(self):
        return self.origHeight

    def getOrigPath(self):
        return self.origPath

    def setSwitch(self, switch):
        self.switch = switch

    def setOk(self, ok):
        self.ok = ok

    def mousePressEvent(self,event):
        if self.switch:
            self.flag = True
            self.x0 = event.x()
            self.y0 = event.y()

    def mouseReleaseEvent(self,event):
        if self.switch:
            self.flag = False

    def mouseMoveEvent(self,event):
        if self.switch:
            if self.flag:
                self.x1 = event.x()
                self.y1 = event.y()
                self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.switch:
            rect =QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
            painter = QPainter(self)
            painter.setPen(QPen(Qt.gray, 1, Qt.SolidLine))
            painter.drawRect(rect)
            #pqscreen  = QGuiApplication.primaryScreen()
            #pixmap2 = pqscreen.grabWindow(1, self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
            #pixmap2.save('0000.png')
            pixmap = QPixmap.copy(self.pixmap, rect)
            pixmap.save('temp/cut.jpg')
            self.ok = True

class newWidget(QWidget):

    def __init__(self, image):
        super(QWidget, self).__init__()
        self.label = newLabel(image)
        #self.resize(self.label.getWidth()+160, self.label.getHeight()+100)
        #self.setMinimumHeight(500)
        self.setFixedSize(960, 830)

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

        #self.propLabel = QLabel()
        #self.propLabel.resize(80, 20)
        #self.propLabel.setText("横纵比")
        #self.propLabel.setAlignment(Qt.AlignLeft)
        #self.propBox = QComboBox()
        #self.propBox.setEditable(False)
        #sizeList = ['原比例', '1:1', '4:3', '3:4', '16:9', '9:16']
        #self.propBox.addItems(sizeList)
        #self.propBox.resize(80, 30)
        #self.propBox.currentIndexChanged.connect(self.propChange)

        self.cutButton = QPushButton('截图')
        self.cutButton.setFixedSize(80, 30)
        self.cutButton.clicked.connect(self.cut)
        self.markButton = QPushButton('添加水印')
        self.markButton.setFixedSize(80, 30)
        self.markButton.clicked.connect(self.mark)
        self.markFlag = True
        self.resetButton = QPushButton('重新设置')
        self.resetButton.setFixedSize(80, 30)
        self.resetButton.clicked.connect(self.reset)
        self.saveButton = QPushButton('保存')
        self.saveButton.setFixedSize(80, 30)
        self.saveButton.clicked.connect(self.save)
        self.shareButton = QPushButton('分享')
        self.shareButton.setFixedSize(80, 30)
        self.shareButton.clicked.connect(self.share)

        self.vboxgroup = QGroupBox()
        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)
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
        #self.vbox.addWidget(self.propLabel)
        #self.vbox.addWidget(self.propBox)
        self.vbox.addWidget(self.cutButton)
        self.vbox.addWidget(self.markButton)
        self.vbox.addWidget(self.resetButton)
        self.vbox.addWidget(self.saveButton)
        self.vbox.addWidget(self.shareButton)
        self.vbox.addStretch(1)
        self.vboxgroup.setLayout(self.vbox)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.label)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.vboxgroup)
        #self.hbox.addStretch(1)
        self.setLayout(self.hbox)

    def change(self, image):
        self.label.changeImage(image)
        #self.resize(self.label.getWidth()+160, self.label.getHeight()+100)
        #self.setMinimumHeight(500)

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

    def propChange(self):
        img_resize(self.label.getImagePath(), self.label.getOrigWidth(), self.label.getOrigHeight(), self.propBox.currentIndex())
        self.change('temp/temp.jpg')
        img_resize(self.label.newImagePath, self.label.getOrigWidth(), self.label.getOrigHeight(), self.propBox.currentIndex(), self.label.newImagePath)

    def cut(self):
        if self.label.switch:
            if self.label.ok:
                self.label.setSwitch(False)
                self.label.setOk(False)
                self.change('temp/cut.jpg')
                shutil.copy('temp/cut.jpg', 'temp/newImage.jpg')
                self.cutButton.setText('截图')
                self.markButton.setText('添加水印')
                self.markFlag = True
                self.updownButton.setDisabled(False)
                self.leftrightButton.setDisabled(False)
                self.clockButton.setDisabled(False)
                self.anticlockButton.setDisabled(False)
                self.brightSlider.setDisabled(False)
                self.sharpSlider.setDisabled(False)
                self.contrastSlider.setDisabled(False)
                #self.propBox.setDisabled(False)
                self.resetButton.setDisabled(False)
                self.saveButton.setDisabled(False)
            else:
                reply = QMessageBox.warning(self, "截图失败", "请选定截图矩形框！", QMessageBox.Ok, QMessageBox.Ok)
        else:
            self.cutButton.setText('确定')
            self.markButton.setText('取消')
            self.markFlag = False
            self.label.setSwitch(True)
            self.updownButton.setDisabled(True)
            self.leftrightButton.setDisabled(True)
            self.clockButton.setDisabled(True)
            self.anticlockButton.setDisabled(True)
            self.brightSlider.setDisabled(True)
            self.sharpSlider.setDisabled(True)
            self.contrastSlider.setDisabled(True)
            #self.propBox.setDisabled(True)
            self.resetButton.setDisabled(True)
            self.saveButton.setDisabled(True)

    def mark(self):
        if self.markFlag:
            text, ok = QInputDialog.getText(self, '添加水印', '请输入：')
            if ok:
                watermark(self.label.getImagePath(), text, "temp/temp.jpg", 1, 20*len(text)+10)
                self.change('temp/temp.jpg')
                watermark(self.label.newImagePath, text, self.label.newImagePath, 1, 20*len(text)+10)
        else:
            self.label.setSwitch(False)
            self.label.setOk(False)
            self.cutButton.setText('截图')
            self.markButton.setText('添加水印')
            self.markFlag = True
            self.change(self.label.getImagePath())
            self.updownButton.setDisabled(False)
            self.leftrightButton.setDisabled(False)
            self.clockButton.setDisabled(False)
            self.anticlockButton.setDisabled(False)
            self.brightSlider.setDisabled(False)
            self.sharpSlider.setDisabled(False)
            self.contrastSlider.setDisabled(False)
            #self.propBox.setDisabled(False)
            self.resetButton.setDisabled(False)
            self.saveButton.setDisabled(False)

    def reset(self):
        self.brightSlider.setValue(50)
        self.sharpSlider.setValue(50)
        self.contrastSlider.setValue(50)
        #self.propBox.setCurrentIndex(0)
        self.label.resetCopy()
        self.change(self.label.getOrigPath())

    def save(self):
        fileName, ok = QFileDialog.getSaveFileName(self, 'save file', './', 'Images ( *.jpg *.png)')
        if ok:
            self.label.pixmap.save(fileName)

    def share(self):
        aFile = open("token", "r")
        aToken = aFile.read()
        aToken = aToken.strip('\n')
        aFile.close()
        if aToken == "0":
            openBrowser()
            code, ok1 = QInputDialog.getText(self, '关联微博账号', '请输入url中的code：')
            if ok1:
                ok2, token = get_token(code)
                if ok2:
                    tokenFile = open("token", "w")
                    tokenFile.write(token)
                    tokenFile.close()
                    text, ok3 = QInputDialog.getText(self, '分享到微博', '请输入您想说的话：')
                    if ok3:
                         post_a_pic(self.label.getImagePath(), token, text)
                else:
                    reply = QMessageBox.warning(self, "错误", "请输入正确的code！", QMessageBox.Ok, QMessageBox.Ok)
        else:
            text, ok3 = QInputDialog.getText(self, '分享到微博', '请输入您想说的话：')
            if ok3:
                post_a_pic(self.label.getImagePath(), aToken, text)
