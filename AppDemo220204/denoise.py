from PyQt5.QtWidgets import QApplication, QSizePolicy, QButtonGroup, QRadioButton, QGroupBox, QStyleOption, QStyle, QWidget, QLabel, QPushButton, QLineEdit, QTableView, QGridLayout, QFileDialog, QMessageBox, QTableWidget, QFrame, QTableWidgetItem
from PyQt5.QtGui import QPainter, QFont
import librosa
from PyQt5 import QtCore
#import pymysql
import datetime
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
from plotTools import Myplot
from denoiseTools import denoise_by_noisereduce

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox, QStyleOption, QStyle, QToolButton, QTextEdit, QFileDialog
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QThread


# class Thread(QThread):
#     def __init__(self):
#         super(Thread, self).__init__()
#
#     def run(self):
#         pass

class denoise(QWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.denoise_layout = QGridLayout()
        self.setLayout(self.denoise_layout)

        # self.label0 = QLabel("1")
        # self.denoise_layout.addWidget(self.label0, 0, 0, 1, 1)

        # self.label1 = QLabel('')
        # self.denoise_layout.addWidget(self.label1, 1, 0, 1, 1)

        self.openfilebutton = QPushButton('导入处理')
        self.denoise_layout.addWidget(self.openfilebutton, 8, 1, 1, 1)
        self.openfilebutton.setObjectName('denoise_button')
        self.openfilebutton.clicked.connect(self.processFile)

        self.rb1 = QRadioButton('noisereduce', self)
        self.denoise_layout.addWidget(self.rb1, 1, 1, 1, 1)
        self.rb2 = QRadioButton('去噪方法2', self)
        self.denoise_layout.addWidget(self.rb2, 4, 1, 1, 1)

        self.bg1 = QButtonGroup(self)
        self.bg1.addButton(self.rb1, 1)
        self.bg1.addButton(self.rb2, 2)
        self.bg1.buttonClicked.connect(self.rbclicked)

        self.rb1.setChecked(True)

        self.denoise_method = 1

        self.ppdcs = QLabel('prop_decrease')
        self.denoise_layout.addWidget(self.ppdcs, 2, 1, 1, 1)
        self.ppdcs.setObjectName('denoise_label')

        self.prop = QLineEdit('0.8')
        self.denoise_layout.addWidget(self.prop, 2, 2, 1, 1)

        self.nstd = QLabel('n_std_thresh_stationary')
        self.denoise_layout.addWidget(self.nstd, 3, 1, 1, 1)
        self.nstd.setObjectName('denoise_label')

        self.thresh = QLineEdit('1.1')
        self.denoise_layout.addWidget(self.thresh, 3, 2, 1, 1)

        self.testlabel = QLabel("just a placeholder" )
        self.denoise_layout.addWidget(self.testlabel, 5, 1, 1, 1)
        self.testlabel.setObjectName('denoise_label')

        self.testline = QLineEdit('blah blah')
        self.denoise_layout.addWidget(self.testline, 5, 2, 1, 1)

        self.Plot_dynamic = QGroupBox()
        self.denoise_layout.addWidget(self.Plot_dynamic, 9, 1, 2, 6)
        self.Plot_dynamic.setObjectName("Plot_dynamic")
        self.fig = Myplot(width=10, height=4, dpi=36)
        self.gridlayout = QGridLayout(self.Plot_dynamic)
        self.gridlayout.addWidget(self.fig)



    def rbclicked(self):
        sender = self.sender()
        if sender == self.bg1:
            if self.bg1.checkedId() == 1:
                self.denoise_method = 1
            elif self.bg1.checkedId() == 2:
                self.denoise_method = 2




    def processFile(self):
    	# 其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        filepath = QFileDialog.getOpenFileName(self, "上传文件", "./")  # 起始路径
        print(filepath)
        if os.path.exists(filepath[0]):
            self.signal, self.rate = librosa.load(filepath[0], sr=48000, offset=None)  # , duration=10)
            data = np.array(self.signal)
            data = data[0: len(self.signal): int(len(self.signal) / 5000) - 1]
            self.ymin = min(data)
            self.ymax = max(data)
            self.t = np.arange(0, len(data))
            self.y = data
            self.fig.axes.cla()
            self.fig.axes.plot(self.t, self.y, '-', color='#438AFE')
            self.fig.axes.set_ylim([self.ymin / 2, self.ymax / 2])
            # self.fig.draw()

            if self.denoise_method == 1:
                savepath = QFileDialog.getSaveFileName(self,
                                                       "保存文件", "./",
                                                       ".wav Files (*.wav)")
                print(self.prop.text())
                print(self.thresh.text())
                self.signal = denoise_by_noisereduce(self.signal, self.rate, savepath[0], prop_decrease=float(self.prop.text()),
                                                     n_std_thresh_stationary=float(self.thresh.text()))
                data = np.array(self.signal)
                data = data[0: len(self.signal): int(len(self.signal) / 5000) - 1]
                self.t = np.arange(0, len(data))
                self.y = data
                self.fig.axes.plot(self.t, self.y, '-', color='red')
                self.fig.draw()
            elif self.denoise_method == 2:
                pass







    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
