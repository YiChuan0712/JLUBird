import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt

import os
import numpy as np
import sys
from plotTools import Myplot
from modelAmerica45 import testClassification

from PyQt5.QtWidgets import QComboBox, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QMessageBox, QStyleOption, QStyle, QToolButton, QTextEdit, QFileDialog
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import QThread






class classify(QWidget):
    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.classify_layout = QGridLayout()
        self.setLayout(self.classify_layout)

        # self.label0 = QLabel("1")
        # self.classify_layout.addWidget(self.label0, 0, 0, 1, 1)

        # self.label1 = QLabel('')
        # self.classify_layout.addWidget(self.label1, 1, 0, 1, 1)

        # 实例化QComBox对象
        self.currentModel = '北美45'
        self.cb = QComboBox(self)
        self.classify_layout.addWidget(self.cb, 0, 0, 1, 1)
        self.cb.setObjectName('classify_combo')
        self.cb.currentIndexChanged[str].connect(self.modelChoice)

        # 单个添加条目
        self.cb.addItem('北美45')
        # 多个添加条目
        self.cb.addItems(['测试1', '测试2', '测试3'])

        self.openfilebutton = QPushButton('导入处理')
        self.classify_layout.addWidget(self.openfilebutton, 0, 1, 1, 1)
        self.openfilebutton.setObjectName('classify_button')
        self.openfilebutton.clicked.connect(self.processFile)

        self.tableWidget = QtWidgets.QTableWidget()
        self.classify_layout.addWidget(self.tableWidget, 1, 0, 1, 1)
        self.tableWidget.setObjectName("tableWidget")





    def modelChoice(self, i):
        self.currentModel = i
        print(self.currentModel)

    def processFile(self):
    	# 其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        filepath = QFileDialog.getOpenFileName(self, "上传文件", "./")  # 起始路径
        print(filepath)
        if os.path.exists(filepath[0]):
            print(filepath[0])
            if self.currentModel == '北美45':
                df = testClassification(filepath[0])
                input_table = df

                input_table_rows = input_table.shape[0]
                input_table_colunms = input_table.shape[1]

                input_table_header = input_table.columns.values.tolist()
                print(input_table_header)

                ###===========读取表格，转换表格，============================================
                ###======================给tablewidget设置行列表头============================

                self.tableWidget.setColumnCount(input_table_colunms)
                self.tableWidget.setRowCount(input_table_rows)
                self.tableWidget.setHorizontalHeaderLabels(input_table_header)

                ###======================给tablewidget设置行列表头============================

                ###================遍历表格每个元素，同时添加到tablewidget中========================
                for i in range(input_table_rows):
                    input_table_rows_values = input_table.iloc[[i]]
                    # print(input_table_rows_values)
                    input_table_rows_values_array = np.array(input_table_rows_values)
                    input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                    # print(input_table_rows_values_list)
                    for j in range(input_table_colunms):
                        input_table_items_list = input_table_rows_values_list[j]
                        # print(input_table_items_list)
                        # print(type(input_table_items_list))

                        ###==============将遍历的元素添加到tablewidget中并显示=======================

                        input_table_items = str(input_table_items_list)
                        newItem = QTableWidgetItem(input_table_items)
                        newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                        self.tableWidget.setItem(i, j, newItem)
        else:
                pass








    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
