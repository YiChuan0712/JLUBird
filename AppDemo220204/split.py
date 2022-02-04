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
from splitTools import split_by_pyAudioAnalysis

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox, QStyleOption, QStyle, QToolButton, QTextEdit, QFileDialog
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QThread


class split(QWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.split_layout = QGridLayout()
        self.setLayout(self.split_layout)

        self.openfilebutton = QPushButton('导入处理')
        self.split_layout.addWidget(self.openfilebutton, 6, 1, 1, 1)
        self.openfilebutton.setObjectName('split_button')
        self.openfilebutton.clicked.connect(self.processFile)

        self.rb1 = QRadioButton('pyAudioAnalysis.audioSegmentation', self)
        self.split_layout.addWidget(self.rb1, 1, 1, 1, 1)
        self.rb2 = QRadioButton('分割方法2', self)
        self.split_layout.addWidget(self.rb2, 4, 1, 1, 1)

        self.bg1 = QButtonGroup(self)
        self.bg1.addButton(self.rb1, 1)
        self.bg1.addButton(self.rb2, 2)
        self.bg1.buttonClicked.connect(self.rbclicked)

        self.rb1.setChecked(True)

        self.split_method = 1

        self.stwin = QLabel('stwin')
        self.split_layout.addWidget(self.stwin, 2, 1, 1, 1)
        self.stwin.setObjectName('split_label')

        self.stwinedit = QLineEdit('5')
        self.split_layout.addWidget(self.stwinedit, 2, 2, 1, 1)

        self.ststep = QLabel('ststep')
        self.split_layout.addWidget(self.ststep, 2, 3, 1, 1)
        self.ststep.setObjectName('split_label')

        self.ststepedit = QLineEdit('0.5')
        self.split_layout.addWidget(self.ststepedit, 2, 4, 1, 1)

        self.smthwd = QLabel('smoothwindow')
        self.split_layout.addWidget(self.smthwd, 3, 1, 1, 1)
        self.smthwd.setObjectName('split_label')

        self.swedit = QLineEdit('0.9')
        self.split_layout.addWidget(self.swedit, 3, 2, 1, 1)

        self.weight = QLabel('weight')
        self.split_layout.addWidget(self.weight, 3, 3, 1, 1)
        self.weight.setObjectName('split_label')

        self.weightedit = QLineEdit('0.2')
        self.split_layout.addWidget(self.weightedit, 3, 4, 1, 1)

        self.test_label2 = QLabel("just a placeholder" )
        self.split_layout.addWidget(self.test_label2, 5, 1, 1, 1)
        self.test_label2.setObjectName('split_label')

        self.test = QLineEdit('blah blah')
        self.split_layout.addWidget(self.test, 5, 2, 1, 1)

        self.Plot_dynamic = QGroupBox()
        self.split_layout.addWidget(self.Plot_dynamic, 7, 1, 2, 6)
        self.Plot_dynamic.setObjectName("Plot_dynamic")
        self.fig = Myplot(width=10, height=4, dpi=36)
        self.gridlayout = QGridLayout(self.Plot_dynamic)
        self.gridlayout.addWidget(self.fig)


    def rbclicked(self):
        sender = self.sender()
        if sender == self.bg1:
            if self.bg1.checkedId() == 1:
                self.split_method = 1
            elif self.bg1.checkedId() == 2:
                self.split_method = 2


    def processFile(self):
    	# 其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        filepath = QFileDialog.getOpenFileName(self, "选取文件夹", "./")  # 起始路径
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

            if self.split_method == 1:
                savepath = QFileDialog.getSaveFileName(self,
                                                       "保存文件", "./",
                                                       ".wav Files (*.wav)")
                print(savepath)
                split_by_pyAudioAnalysis(filepath[0], savepath[0], float(self.stwinedit.text()),
                                         float(self.ststepedit.text()),
                                         float(self.swedit.text()),
                                         float(self.weightedit.text()))
            self.fig.draw()









    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
