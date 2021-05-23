import cv2
import myGui
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
import numpy as np


class Gui(QtWidgets.QMainWindow, myGui.Ui_MainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        self.setupUi(self)
        self.btn_browse.clicked.connect(self.browse)
        self.rdbtn_negative.toggled.connect(self.rdbtn_negative_clicked)
        self.rdbtn_threshold.toggled.connect(self.rdbtn_threshold_clicked)
        self.rdbtn_gradient.toggled.connect(self.rdbtn_gradient_clicked)
        self.rdbtn_linearScaling.toggled.connect(self.rdbtn_linearScaling_clicked)
        self.spbox_threshold.valueChanged.connect(self.rdbtn_threshold_clicked)
        self.spbox_gradient.valueChanged.connect(self.rdbtn_gradient_clicked)

        self.originalImage = None
        self.greyImage = None

        self.rdbtn_negative.setEnabled(False)
        self.rdbtn_linearScaling.setEnabled(False)
        self.rdbtn_histogramEqualization.setEnabled(False)
        self.rdbtn_threshold.setEnabled(False)
        self.rdbtn_gradient.setEnabled(False)
        self.spbox_threshold.setEnabled(False)
        self.spbox_gradient.setEnabled(False)

        self.fig, self.ax = plt.subplots()

    def browse(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "~/", "Images (*.png *.xpm *.jpg)")
        if filename:

            self.originalImage = cv2.imread(filename, -1)
            self.greyImage = cv2.imread(filename, 0)

            pixmap = QPixmap(filename)
            my_scaled_pixmap = pixmap.scaled(self.lbl_originalImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_originalImage.setPixmap(my_scaled_pixmap)

            image = QtGui.QImage(self.greyImage, self.greyImage.shape[1], self.greyImage.shape[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QPixmap(image)
            my_scaled_pixmap = pixmap.scaled(self.lbl_originalImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_greyImage.setPixmap(my_scaled_pixmap)

            self.rdbtn_negative.setEnabled(True)
            self.rdbtn_linearScaling.setEnabled(True)
            self.rdbtn_histogramEqualization.setEnabled(False)
            self.rdbtn_threshold.setEnabled(True)
            self.rdbtn_gradient.setEnabled(True)

            self.rdbtn_negative.setAutoExclusive(False)
            self.rdbtn_negative.setChecked(False)
            self.rdbtn_negative.setAutoExclusive(True)

            self.rdbtn_linearScaling.setAutoExclusive(False)
            self.rdbtn_linearScaling.setChecked(False)
            self.rdbtn_linearScaling.setAutoExclusive(True)

            self.rdbtn_histogramEqualization.setAutoExclusive(False)
            self.rdbtn_histogramEqualization.setChecked(False)
            self.rdbtn_histogramEqualization.setAutoExclusive(True)

            self.rdbtn_threshold.setAutoExclusive(False)
            self.rdbtn_threshold.setChecked(False)
            self.rdbtn_threshold.setAutoExclusive(True)

            self.rdbtn_gradient.setAutoExclusive(False)
            self.rdbtn_gradient.setChecked(False)
            self.rdbtn_gradient.setAutoExclusive(True)


            self.spbox_threshold.setValue(0)
            self.spbox_gradient.setValue(1)

            self.wdgt_histblue.clear()
            data, _ = np.histogram(self.originalImage[:, :, 0].flatten(), bins=256, normed=True)
            self.wdgt_histblue.plotItem.plot(data)

            self.wdgt_histgreen.clear()
            data, _ = np.histogram(self.originalImage[:, :, 1].flatten(), bins=256, normed=True)
            self.wdgt_histgreen.plotItem.plot(data)

            self.wdgt_histred.clear()
            data, _ = np.histogram(self.originalImage[:, :, 2].flatten(), bins=256, normed=True)
            self.wdgt_histred.plotItem.plot(data)

            self.wdgt_histgrey.clear()
            data, _ = np.histogram(self.greyImage.flatten(), bins=256, normed=True)
            self.wdgt_histgrey.plotItem.plot(data)

            self.wdgt_histmodifiedImage.clear()
            self.lbl_modifiedImage.setText('Modified Image')


    # Negation
    def rdbtn_negative_clicked(self):
        if self.rdbtn_negative.isChecked():
            self.spbox_threshold.setEnabled(False)
            self.spbox_gradient.setEnabled(False)

            full255 = np.full(self.greyImage.shape, 255, self.greyImage.dtype)

            negative_image = full255 - self.greyImage
            image = QtGui.QImage(negative_image, self.greyImage.shape[1], self.greyImage.shape[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QPixmap(image)
            my_scaled_pixmap = pixmap.scaled(self.lbl_modifiedImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_modifiedImage.setPixmap(my_scaled_pixmap)

            self.wdgt_histmodifiedImage.clear()
            data, _ = np.histogram(negative_image.flatten(), bins=256, normed=True)
            self.wdgt_histmodifiedImage.plotItem.plot(data)

    # Linear Scaling
    def rdbtn_linearScaling_clicked(self):
        if self.rdbtn_linearScaling.isChecked():
            self.spbox_threshold.setEnabled(False)
            self.spbox_gradient.setEnabled(False)

            intensity_range = np.amax(self.greyImage) - np.amin(self.greyImage)
            stretched_image = (self.greyImage - np.full(self.greyImage.shape, np.amin(self.greyImage),
                                                        self.greyImage.dtype)) / intensity_range * np.amax(self.greyImage)
            stretched_image = np.asanyarray(stretched_image, dtype=self.greyImage.dtype)

            self.wdgt_histmodifiedImage.clear()
            data, _ = np.histogram(stretched_image.flatten(), bins=256, normed=True)
            self.wdgt_histmodifiedImage.plotItem.plot(data)

            image = QtGui.QImage(stretched_image, self.greyImage.shape[1], self.greyImage.shape[0],
                                 QtGui.QImage.Format_Grayscale8)
            pixmap = QPixmap(image)
            my_scaled_pixmap = pixmap.scaled(self.lbl_modifiedImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_modifiedImage.setPixmap(my_scaled_pixmap)

    # histogram Equalization #not implemented yet
    def rdbtn_histofgramEqualization_clicked(self):
        if rdbtn_histofgramEqualization.isChecked():
            pass

    # Threshold
    def rdbtn_threshold_clicked(self):
        if self.rdbtn_threshold.isChecked():
            self.spbox_gradient.setEnabled(False)
            self.spbox_threshold.setEnabled(True)
            threshold = self.spbox_threshold.value()
            threshold_image = np.zeros(self.greyImage.shape, dtype=self.greyImage.dtype)

            threshold_image[self.greyImage < threshold] = 0
            threshold_image[self.greyImage >= threshold] = 255

            self.wdgt_histmodifiedImage.clear()
            data, _ = np.histogram(threshold_image.flatten(), bins=256, normed=True)
            self.wdgt_histmodifiedImage.plotItem.plot(data)

            image = QtGui.QImage(threshold_image, self.greyImage.shape[1], self.greyImage.shape[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QPixmap(image)
            my_scaled_pixmap = pixmap.scaled(self.lbl_modifiedImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_modifiedImage.setPixmap(my_scaled_pixmap)

    # Gradient
    def rdbtn_gradient_clicked(self):
        if self.rdbtn_gradient.isChecked():
            self.spbox_threshold.setEnabled(False)
            self.spbox_gradient.setEnabled(True)

            brightness = self.spbox_gradient.value()

            image_x = np.roll(self.greyImage, 1, 0) - self.greyImage
            image_y = np.roll(self.greyImage, 1, 1) - self.greyImage

            gradient_image = ((image_x**2 + image_y**2)**0.5)
            gradient_image = np.asanyarray(gradient_image, dtype=self.greyImage.dtype) * brightness

            self.wdgt_histmodifiedImage.clear()
            data, _ = np.histogram(gradient_image.flatten(), bins=256, normed=True)
            self.wdgt_histmodifiedImage.plotItem.plot(data)

            image = QtGui.QImage(gradient_image, self.greyImage.shape[1], self.greyImage.shape[0], QtGui.QImage.Format_Grayscale8)
            pixmap = QPixmap(image)
            my_scaled_pixmap = pixmap.scaled(self.lbl_modifiedImage.size(), QtCore.Qt.KeepAspectRatio)
            self.lbl_modifiedImage.setPixmap(my_scaled_pixmap)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Gui()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()