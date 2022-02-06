from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QStyleOption, QStyle, QLabel
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter


class leftNavigation(QWidget):
    '''
    左侧导航栏
    '''

    def __init__(self):
        super().__init__()
        self.init()

    def init(self):
        self.left_layout = QGridLayout()  # 创建网格布局对象
        self.setLayout(self.left_layout)  # 将左侧部件设置为网格布局

        # 初始化创建左侧标签和按钮
        self.init_left_label()
        self.init_left_operation()

        self.left_layout.addWidget(self.left_button0, 0, 0, 1, 3)
        self.left_layout.addWidget(self.left_label1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button3, 4, 0, 1, 3)

        self.left_layout.addWidget(self.left_label2, 5, 0, 1, 3)
        self.left_layout.addWidget(self.left_button4, 6, 0, 1, 3)
        self.left_layout.addWidget(self.left_button5, 7, 0, 1, 3)

        self.left_layout.addWidget(self.left_label3, 8, 0, 1, 3)
        self.label1 = QLabel()  # 用来支撑空间
        self.left_layout.addWidget(self.label1, 9, 0, 1, 3)



    def init_left_label(self):
        '''
        左侧标题栏
        '''
        self.left_label1 = QPushButton('音频处理')
        self.left_label1.setObjectName('left_label')
        self.left_label2 = QPushButton('测试')
        self.left_label2.setObjectName('left_label')
        self.left_label3 = QPushButton('联系与帮助')
        self.left_label3.setObjectName('left_label')

    def init_left_operation(self):
        '''
        左侧操作按钮
        '''
        self.left_button0 = QPushButton(' 欢迎界面')
        self.left_button0.setObjectName('left_button')
        #self.left_button1.setIcon(QtGui.QIcon('./img/bookquery.png'))
        self.left_button0.setCheckable(True)
        self.left_button0.setAutoExclusive(True)
        self.left_button0.setChecked(True)

        self.left_button1 = QPushButton(' 鸟声识别')
        self.left_button1.setObjectName('left_button')
        #self.left_button1.setIcon(QtGui.QIcon('./img/bookquery.png'))
        self.left_button1.setCheckable(True)
        self.left_button1.setAutoExclusive(True)

        self.left_button2 = QPushButton(' 音频去噪')
        self.left_button2.setObjectName('left_button')
        #self.left_button2.setIcon(QtGui.QIcon('./img/bookborrow.png'))
        self.left_button2.setCheckable(True)
        self.left_button2.setAutoExclusive(True)

        self.left_button3 = QPushButton(' 音频分割')
        self.left_button3.setObjectName('left_button')
        #self.left_button3.setIcon(QtGui.QIcon('./img/bookreturn.png'))
        self.left_button3.setCheckable(True)
        self.left_button3.setAutoExclusive(True)

        self.left_button4 = QPushButton(' 测试')
        self.left_button4.setObjectName('left_button')
        #self.left_button4.setIcon(QtGui.QIcon('./img/bookin.png'))
        self.left_button4.setCheckable(True)
        self.left_button4.setAutoExclusive(True)

        self.left_button5 = QPushButton(' 测试')
        self.left_button5.setObjectName('left_button')
        #self.left_button5.setIcon(QtGui.QIcon('./img/bookcard.png'))
        self.left_button5.setCheckable(True)
        self.left_button5.setAutoExclusive(True)

    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
