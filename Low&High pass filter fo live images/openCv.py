import gui
import cv2
import numpy as np
from matplotlib import pyplot as plt

from PyQt5 import QtWidgets, QtCore, QtGui
#from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QApplication
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas)
import sys
import os


class Gui(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        self.setupUi(self)
        self.btnBrowse.clicked.connect(self.browse)
        self.btnCameraOpen.clicked.connect(self.openCamera)
        self.btnCameraClose.clicked.connect(self.closeCamera)

        self.closeFlag = False
        self.recordFlag = False
        self.camToggleFlag = True
        self.cap = None
        self.frame = None
        self.originalImg = None
        self.lowPassImg = None
        self.highPassImg = None
        self.myPath = './temp'
        self.btnCapture.clicked.connect(self.capture)
        self.btnToggle.clicked.connect(self.camToggle)
        self.btnReset.clicked.connect(self.clear)
        self.actionSave_Image.triggered.connect(self.saveOriginal)
        self.actionSave_Low_Pass_Image.triggered.connect(self.saveLowPassImage)
        self.actionSave_High_Pass_Image.triggered.connect(self.saveHighPassImage)
        self.actionExit.triggered.connect(self.exitApp)
        self.onStart()

    # Warning for unsupported extension on saving images
    def msgBox(self):
        QtWidgets.QMessageBox.about(self, "Unsupport Extension", "Unsupported Image Extension")

    # create a hidden folder to store the image
    def onStart(self):
        if not os.path.exists(self.myPath):
            os.makedirs(self.myPath)
            os.chmod(self.myPath, 0o777)
        self.statusbar.showMessage('Select either to browse or open camera')

    # browse for an image
    def browse(self):
        self.statusbar.clearMessage()
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "~/", "Images (*.png *.xpm *.jpg)")
        if filename:
            self.filter(filename)

    # display Image and apply blur filter "low pass" and canny filter "high pass"
    def filter(self, filename):

        highPassImgStr = './temp/high.png'
        lowPassImgStr = './temp/low.png'


        # display original Image on the first label
        self.originalImg = cv2.imread(filename, -1)

        pixmap = QPixmap(filename)
        myScaledPixmap = pixmap.scaled(self.lblImage.size(), QtCore.Qt.KeepAspectRatio)
        self.lblImage.setPixmap(myScaledPixmap)

        # display Low Pass Image on the second label

        self.lowPassImg = cv2.blur(self.originalImg, (5, 5))
        cv2.imwrite(lowPassImgStr, self.lowPassImg)

        pixmap = QPixmap(lowPassImgStr)
        myScaledPixmap = pixmap.scaled(self.lblLowPass.size(), QtCore.Qt.KeepAspectRatio)
        self.lblLowPass.setPixmap(myScaledPixmap)

        # display High Pass Image on the third label
        self.highPassImg = cv2.Canny(self.originalImg,100,200)
        cv2.imwrite(highPassImgStr, self.highPassImg)

        pixmap = QPixmap(highPassImgStr)
        myScaledPixmap = pixmap.scaled(self.lblHighPass.size(), QtCore.Qt.KeepAspectRatio)
        self.lblHighPass.setPixmap(myScaledPixmap)

    # open the camera
    def openCamera(self):

        self.closeFlag = False
        self.cap = cv2.VideoCapture(0)

        while self.cap.isOpened():
            self.statusbar.showMessage('Smile :)')
            self.recordFlag = True
            ret, self.frame = self.cap.read()

            if self.camToggleFlag:
                cv2.destroyAllWindows()
                self.btnToggle.setText("Camera out of window")
                cv2.imwrite('./temp/frame.png', self.frame)
                pixmap = QPixmap('./temp/frame.png')
                myScaledPixmap = pixmap.scaled(self.lblVideo.size(), QtCore.Qt.KeepAspectRatio)
                self.lblVideo.setPixmap(myScaledPixmap)
            else:
                self.lblVideo.setText('Video Panel')
                self.btnToggle.setText("Camera in window")
                cv2.imshow('frame', self.frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')) or self.closeFlag:
                self.statusbar.clearMessage()
                self.recordFlag = False
                break
            elif cv2.waitKey(20) & 0xFF == ord('y'):
                self.capture()


        self.cap.release()
        cv2.destroyAllWindows()
        self.lblVideo.setText('Video Panel')


    # close the camera
    def closeCamera(self):
        self.statusbar.clearMessage()
        self.recordFlag = False
        self.closeFlag = True

    # save a frame then display it and apply the filters
    def capture(self):
        if self.recordFlag:
            filename = './temp/CamImage.png'
            cv2.imwrite(filename, self.frame)
            self.filter(filename)
        else:
            self.statusbar.showMessage('Please press on Open Camera Button')

    # camera in or out window
    def camToggle(self):
        self.camToggleFlag = not self.camToggleFlag

    # menu bar :: save original image
    def saveOriginal(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.png', 'Images (*.png *.xpm *.jpg)')[0]
        if path:
            if path.endswith(('.png', '.jpg', '.xpm')):
                cv2.imwrite(path, self.originalImg)
            elif '.' in path:
                self.msgBox()
            else:
                path += '.png'
                cv2.imwrite(path, self.originalImg)

    # menu bar :: save low pass image
    def saveLowPassImage(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.png', 'Images (*.png *.xpm *.jpg)')[0]
        if path:
            if path.endswith(('.png', '.jpg', '.xpm')):
                cv2.imwrite(path, self.lowPassImg)
            elif '.' in path:
                self.msgBox()

            else:
                path += '.png'
                cv2.imwrite(path, self.lowPassImg)

    # menu bar :: save high pass image
    def saveHighPassImage(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.png', 'Images (*.png *.xpm *.jpg)')[0]
        if path:
            if path.endswith(('.png', '.jpg', '.xpm')):
                cv2.imwrite(path, self.highPassImg)
            elif '.' in path:
                self.msgBox()
            else:
                path += '.png'
                cv2.imwrite(path, self.highPassImg)

    # menu bar :: close app
    def exitApp(self):
        sys.exit()

    def clear(self):
        self.lblImage.setText('Original Image')
        self.lblLowPass.setText('Low Pass Filtered Image')
        self.lblHighPass.setText('High Pass Filtered Image')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Gui()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
