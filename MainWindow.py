import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QTabWidget, QMainWindow, QApplication
import PreDefined
import UserDefined

class myTabWidget(QTabWidget):
    
    def __init__(self):
        super(QTabWidget, self).__init__()
        self.initUI()

    def initUI(self):
        self.preDefined = PreDefined.myPreDefined()
        self.userDefined = UserDefined.myUserDefined()
        self.addTab(self.preDefined, 'Predefined')
        self.addTab(self.userDefined, 'User Defined')

class myMainWindow(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.move(300, 300)
        self.resize(900, 800)
        self.setWindowTitle('Image Style Transfer')
        self.setCentralWidget(myTabWidget())

if __name__ == '__main__':
    with open("template.qss") as file:
        style = file.readlines()
        style = "".join(style).strip("\n")
    app = QApplication(sys.argv)
    app.setStyleSheet(style)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())
