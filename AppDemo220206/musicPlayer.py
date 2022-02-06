from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QStyleOption, QStyle, QLabel, QListWidget, QSlider
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt
import threading

from PyQt5.QtCore import QUrl, QFileInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist

import os
import sys

class musicPlayer(QWidget):

    player = QMediaPlayer()
    playlist = QMediaPlaylist()
    playing = False

    def __init__(self):

        super().__init__()
        self.init()

    def init(self):
        self.music_layout = QGridLayout()  # 创建网格布局对象
        self.setLayout(self.music_layout)  # 将左侧部件设置为网格布局

        # 初始化创建左侧标签和按钮
        # self.init_music_label()
        self.init_music_operation()

        self.music_layout.addWidget(self.openfile, 0, 0, 1, 5)
        # self.music_layout.addWidget(self.label_2, 1, 0, 1, 1)
        self.music_layout.addWidget(self.nowtime, 1, 0, 1, 1)
        self.music_layout.addWidget(self.progressSlider, 1, 1, 1, 3)
        self.music_layout.addWidget(self.duration, 1, 4, 1, 1)
        self.music_layout.addWidget(self.skipPrev, 2, 0, 1, 1)
        self.music_layout.addWidget(self.playButton, 2, 1, 1, 1)
        self.music_layout.addWidget(self.skipNext, 2, 2, 1, 1)
        self.music_layout.addWidget(self.label, 2, 3, 1, 1)
        self.music_layout.addWidget(self.volumeSlider, 2, 4, 1, 1)
        # self.music_layout.addWidget(self.label_4, 3, 0, 1, 1)
        self.music_layout.addWidget(self.playlist_UI, 4, 0, 4, 5)

        self.label1 = QLabel()  # 用来支撑空间
        self.music_layout.addWidget(self.label1, 9, 0, 1, 5)

        self.volumeSlider.valueChanged.connect(self.volumeSliderChanged)
        self.progressSlider.valueChanged.connect(self.player.setPosition)
        self.openfile.clicked.connect(self.openMediaFilesDialog)

        self.playButton.clicked.connect(self.play)
        self.skipNext.clicked.connect(self.skip_next)
        self.skipPrev.clicked.connect(self.skip_prev)
        self.player.error.connect(self.player_error)
        self.player.durationChanged.connect(self.setMax)
        self.player.positionChanged.connect(self.moveSlider)

        self.thread = threading.Thread(target=self.play_counter)
        self.thread.start()
        self.player.setPlaylist(self.playlist)




    def init_music_operation(self):
        '''
        操作按钮
        '''
        self.music_button0 = QPushButton('按钮')
        self.music_button0.setObjectName('music_button')
        #self.music_button1.setIcon(QtGui.QIcon('./img/bookquery.png'))
        self.music_button0.setCheckable(True)
        self.music_button0.setAutoExclusive(True)
        self.music_button0.setChecked(True)

        self.label_4 = QLabel('Playlist')
        self.label_4.setObjectName("label_4")

        self.label = QLabel('Volume')
        self.label.setObjectName("label")

        self.nowtime = QLabel('00:00:00')
        self.nowtime.setObjectName("nowtime")

        self.duration = QLabel('00:00:00')
        self.duration.setObjectName("duration")

        self.openfile = QPushButton('openfile(仅在测试阶段使用)')
        self.openfile.setObjectName("openfile")

        self.playlist_UI = QListWidget()
        self.playlist_UI.setObjectName("playlist_UI")

        self.progressSlider = QSlider()
        self.progressSlider.setSliderPosition(0)
        self.progressSlider.setOrientation(Qt.Horizontal)
        self.progressSlider.setObjectName("progressSlider")

        self.skipPrev = QPushButton("<")
        self.skipPrev.setObjectName("skipPrev")

        self.volumeSlider = QSlider()
        self.volumeSlider.setOrientation(Qt.Horizontal)
        self.volumeSlider.setObjectName("volumeSlider")

        self.playButton = QPushButton("Play")
        self.playButton.setObjectName("playButton")

        self.label_2 = QLabel('Progress')
        self.label_2.setObjectName("label_2")

        self.skipNext = QPushButton(">")
        self.skipNext.setObjectName("skipNext")

    def setMax(self, dur):
        print("DurationChanged: ", dur)
        self.progressSlider.setMaximum(dur)
        hour = int((dur / 1000) / 3600)
        mnt = int((dur / 1000) / 60)
        sec = int(dur / 1000) - hour * 3600 - mnt * 60
        self.duration.setText("%02d:%02d:%02d" % (hour, mnt, sec))

    def moveSlider(self, position):
        print("PositionChanged: ", position, self.player.duration())
        self.progressSlider.setMaximum(self.player.duration())
        self.progressSlider.setValue(position)
        hour = int((position / 1000) / 3600)
        mnt = int((position / 1000) / 60)
        sec = int(position / 1000) - hour * 3600 - mnt * 60
        self.nowtime.setText("%02d:%02d:%02d" % (hour, mnt, sec))

    def play(self):
        if not self.playing:
            if self.player.currentMedia().isNull():
                self.skip_next()
            else:
                self.player.play()

            self.playing = True
        elif self.playing:
            self.player.pause()
            self.playing = False

    def open_files(self, files):
        for f in files:
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(f)))
            self.playlist_UI.addItem(os.path.basename(f))

    def skip_prev(self):
        if self.playlist.currentMedia().isNull():
            pass
        else:
            self.playlist.prev()

    def skip_next(self):
        if self.playlist.currentMedia().isNull():
            self.playlist.setCurrentIndex(1)
            self.player.play()
        else:
            self.playlist.next()

    def volumeSliderChanged(self):
        self.player.setVolume(self.volumeSlider.value())

    def play_counter(self):
        while True:
            if (self.playing):
                self.playButton.setText("Pause")
            else:
                self.playButton.setText("Play")

    def track_slider(self):
        print(self.player.position(), self.player.duration())

    def openMediaFilesDialog(self):
        # options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Media Files", "", "All Files (*)")  # , options=options)
        # files = QFileDialog.getOpenFileNames(self, "选取文件夹", "./")[0]  # 起始路径
        print(files)
        if files:
            self.open_files(files)

    def player_error(self):
        print(self.player.error())

    def paintEvent(self, event):
        '''
        避免多重传值后的功能失效，从而可以继续使用qss设置样式
        '''
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
