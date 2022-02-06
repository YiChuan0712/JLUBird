from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QStyleOption, QStyle, QLabel
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt


class topBar(QWidget):
    '''
    左侧导航栏
    '''

    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.top_layout = QGridLayout()  # 创建网格布局对象
        self.setLayout(self.top_layout)  # 将左侧部件设置为网格布局

        # 初始化创建按钮
        self.init_top_close_mini_visit()

        # 将初始化完成的左侧标签、按钮添加进左侧网格布局
        # 最小化、放大还原、关闭部分按钮
        self.label1 = QLabel()
        self.top_layout.addWidget(self.top_logo, 0, 0, 1, 1)
        self.top_layout.addWidget(self.label1, 0, 1, 1, 1)
        self.top_layout.addWidget(self.top_mini, 0, 2, 1, 1)
        self.top_layout.addWidget(self.top_visit, 0, 3, 1, 1)
        self.top_layout.addWidget(self.top_close, 0, 4, 1, 1)


    def init_top_close_mini_visit(self):
        '''
        创建关闭、最小化、放大还原按钮
        '''
        self.top_logo = QPushButton("吉大听鸟")  # 最小化按钮
        self.top_logo.setObjectName('top_logo')

        self.top_close = QPushButton("×")  # 关闭按钮
        self.top_close.setObjectName('top_close')
        self.top_close.setToolTip("关闭")

        self.top_mini = QPushButton("-")  # 最小化按钮
        self.top_mini.setObjectName('top_mini')
        self.top_mini.setToolTip("最小化")

        self.top_visit = QPushButton("□")  # 空白按钮
        self.top_visit.setObjectName('top_visit')
        self.top_visit.setToolTip("全屏/自适应")



    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
