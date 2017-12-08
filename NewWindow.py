from PyQt5.QtWidgets import QLabel, QWidget
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication, QPainter, QPen
from PyQt5.QtCore import QRect
from PyQt5.Qt import Qt

class newLabel(QLabel):

    def __init__(self, parent):
        super(QLabel, self).__init__(parent)
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.flag = False

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
        pqscreen  = QGuiApplication.primaryScreen()
        pixmap2 = pqscreen.grabWindow(self.winId(), self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        pixmap2.save('0000.png')

class newWidget(QWidget):

    def __init__(self, image):
        super(QWidget, self).__init__()
        self.setFixedSize(image.width(), image.height())
        self.label = newLabel(self)
        self.label.setFixedSize(image.width(), image.height())
        self.label.setPixmap(QPixmap.fromImage(image))
