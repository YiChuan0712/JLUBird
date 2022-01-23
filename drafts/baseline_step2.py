"""
基线 - 第二步
音频转为图像（语谱图）
"""
import os
import warnings
from sklearn.model_selection import train_test_split
import shutil
warnings.filterwarnings(action='ignore')

filename = open('im.txt', 'r')
str_samples = filename.read()
filename.close()
str_samples = str_samples.replace("\\", "/")
samples = str_samples.split(',')
trainval_files, test_files = train_test_split(samples, test_size=0.3, random_state=42)
train_dir = r'D:/working/train/'
val_dir = r'D:/working/val/'


def copyfiles(file, dir):
    filelist = file.split('/')
    filename = filelist[-1]
    lable = filelist[-2]
    cpfile = dir + "/" + lable
    if not os.path.exists(cpfile):
        os.makedirs(cpfile)
    cppath = cpfile + '/' + filename
    shutil.copy(file, cppath)


for file in trainval_files:
    copyfiles(file, train_dir)
for file in test_files:
    copyfiles(file, val_dir)
