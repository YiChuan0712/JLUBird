from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QPushButton, QDialog, QFrame, QLabel, QToolButton, QFileDialog
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtCore import Qt
import sys

from leftNavigation import leftNavigation
from topBar import topBar
from musicPlayer import musicPlayer

from denoise import denoise
from split import split
from classify import classify


class mainSystem(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        '''
        初始化整体布局
        '''
        self.desktopWidth = QApplication.desktop().width()  # 获取当前桌面的宽
        self.desktopHeight = QApplication.desktop().height()  # 获取当前桌面的高
        self.resize(self.desktopWidth * 0.695, self.desktopHeight * 0.695)

        self.main_widget = QWidget()  # 创建窗口主部件
        self.main_widget.setObjectName('main_widget')  # 对象命名
        self.main_layout = QGridLayout()  # 创建网格布局的对象
        self.main_widget.setLayout(self.main_layout)  # 将主部件设置为网格布局

        self.init_left()  # 初始化左侧空间
        self.init_right()  # 初始化右侧空间
        self.init_top()  # 初始化顶部条
        self.init_music()

        self.main_layout.addWidget(self.top_widget, 0, 0, 1, 8)
        self.main_layout.addWidget(self.left_widget, 1, 0, 12, 1)
        self.main_layout.addWidget(self.right_widget, 1, 1, 12, 5)
        self.main_layout.addWidget(self.music_widget, 1, 6, 12, 2)

        self.main_layout.addWidget(self.denoise_widget, 1, 1, 12, 5)

        self.main_layout.addWidget(self.split_widget, 1, 1, 12, 5)

        self.main_layout.addWidget(self.classify_widget, 1, 1, 12, 5)

        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

        # 窗口属性设置
        self.setWindowOpacity(1)  # 设置窗口透明度
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏边框
        self.main_layout.setSpacing(0)  # 取出左右之间的缝隙

    def init_music(self):
        self.music_widget = musicPlayer()
        self.music_widget.setObjectName('music_widget')

    def init_left(self):
        '''
        初始化左侧布局
        '''
        self.left_widget = leftNavigation()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')  # 左侧部件对象命名
        self.left_widget.left_button0.clicked.connect(self.into_mainView)
        self.left_widget.left_button1.clicked.connect(self.into_classifyView)
        self.left_widget.left_button2.clicked.connect(self.into_denoiseView)
        self.left_widget.left_button3.clicked.connect(self.into_splitView)

    def init_right(self):
        '''
        初始化右侧布局
        '''
        self.main_view()
        self.denoise_view()
        self.split_view()
        self.classify_view()

    def init_top(self):

        self.top_widget = topBar()  # 创建
        self.top_widget.setObjectName('top_widget')  # 部件对象命名

        self.top_widget.top_close.clicked.connect(self.closeWindow)
        self.top_widget.top_mini.clicked.connect(self.minimizeWindow)
        self.visitFlag = False
        self.top_widget.top_visit.clicked.connect(self.visitWindow)



    def main_view(self):
        '''
        用于介绍的主界面
        '''

        self.right_widget = QWidget()  # 创建右侧界面1
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QGridLayout()  # 创建网格布局对象1
        self.right_widget.setLayout(self.right_layout)  # 设置右侧界面1的布局为网格布局

        # 支撑空间
        self.label1 = QLabel()
        self.right_layout.addWidget(self.label1, 0, 0, 1, 4)

        introduction = '''
            <div style="text-align:center; font-size: 60px;">
            <b>欢迎使用吉大听鸟</b></div> 
            <div style="text-align:center; font-size: 25px;">
            <br>
            <b>蓝色 #438AFE</b></div>
            <div style="text-align:center; font-size: 25px;">
            <b>测试</b></div>
            <div style="text-align:center; font-size: 25px;">
            <b>测试</b></div>
        '''

        self.label_introduction = QLabel()
        self.label_introduction.setText(introduction)
        self.label_introduction.setObjectName('introduction')
        self.right_layout.addWidget(self.label_introduction, 3, 1, 2, 2)

        self.noneLabel1_2 = QLabel()  # 用来支撑空间
        self.right_layout.addWidget(self.noneLabel1_2, 5, 0, 2, 4)

    def denoise_view(self):
        '''
        借书界面
        '''

        self.denoise_widget = denoise()
        self.denoise_widget.setObjectName('denoise_widget')
        self.denoise_widget.hide()

    def split_view(self):
        '''
        还书界面
        '''
        self.split_widget = split()
        self.split_widget.setObjectName('split_widget')
        self.split_widget.hide()

    def classify_view(self):
        self.classify_widget = classify()
        self.classify_widget.setObjectName('classify_widget')
        self.classify_widget.hide()

    def into_mainView(self):
        '''
        切换到借书界面
        '''
        self.right_widget.show()
        self.denoise_widget.hide()
        self.split_widget.hide()
        self.classify_widget.hide()

    def into_denoiseView(self):
        '''
        切换到借书界面
        '''
        self.denoise_widget.show()
        self.right_widget.hide()
        self.split_widget.hide()
        self.classify_widget.hide()

    def into_splitView(self):
        '''
        切换到还书界面
        '''
        self.split_widget.show()
        self.denoise_widget.hide()
        self.right_widget.hide()
        self.classify_widget.hide()

    def into_classifyView(self):
        '''
        切换到还书界面
        '''
        self.classify_widget.show()
        self.split_widget.hide()
        self.denoise_widget.hide()
        self.right_widget.hide()


    def closeWindow(self):
        '''
        close按钮对应的关闭窗口
        '''
        self.close()

    def minimizeWindow(self):
        '''
        mini按钮对应的最小化窗口
        '''
        self.showMinimized()

    def visitWindow(self):
        '''
        visit按钮对应的全屏or还原窗口大小
        '''
        if self.visitFlag == False:
            self.lastWidth = self.width()
            self.lastHeight = self.height()
            self.resize(self.desktopWidth, self.desktopHeight)
            x = (self.desktopWidth - self.width()) // 2
            y = (self.desktopHeight - self.height()) // 2
            self.move(x, y)
            # print('max')
            self.visitFlag = True
        else:
            self.resize(self.lastWidth, self.lastHeight)
            x = (self.desktopWidth - self.width()) // 2
            y = (self.desktopHeight - self.height()) // 2
            self.move(x, y)
            # print('origin')
            self.visitFlag = False

    def mousePressEvent(self, QMouseEvent):
        '''
        redefine已有的鼠标按下事件
        '''
        if QMouseEvent.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = QMouseEvent.globalPos()-self.pos()  # 获取鼠标相对窗口的位置
            QMouseEvent.accept()
            # self.setCursor(QCursor(Qt.WaitCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):
        '''
        redefine已有的鼠标移动事件
        '''
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos()-self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        '''
        redefine已有的鼠标释放事件
        '''
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


def main():
    with open('app.qss', encoding='utf-8') as f:
        qss = f.read()
    app = QApplication(sys.argv)
    app.setStyleSheet(qss)
    gui = mainSystem()
    # gui.setWindowIcon(QIcon('./img/xxx.ico'))
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
