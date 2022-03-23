"""
对audio_images中的mel图进行鸟声检测

输入
    ../input/kkiller-birdclef-mels-computer-d7-part/audio_images/        - mel图片
    ../input/birdclef-2021/train_metadata.csv        - 元数据
    ../input/clef-nocall-2class2-5fold/        - 鸟声检测器

输出
    ./nocalldetection_for_shortaudio_fold{n}.csv        - 识别结果，{n}与检测器对应

    将输出整理到 ../input/train_short_audio_nocall_fold0to4/
"""

import os
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations.pytorch.transforms import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

import warnings
warnings.filterwarnings('ignore')

import glob

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class CFG:
    debug = False
    print_freq = 100
    num_workers = 0
    model_name = 'resnext50_32x4d'
    dim = (128, 281)
    epochs = 10
    batch_size = 1
    seed = 42
    target_size = 2
    fold = 0  # choose from [0,1,2,3,4]
    pretrained = False
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')


def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed=CFG.seed)


short = pd.read_csv('../input/birdclef-2021/train_metadata.csv')

# short 送入此类
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.filenames = df['filename'].values
        # self.labels = df['hasbird'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        filepath = glob.glob(f'../input/kkiller-birdclef-mels-computer-d7-part/audio_images/*/{file_name}.npy')[0]
        image = np.load(filepath)
        image = np.stack((image,) * 3, -1)
        augmented_images = []
        if self.transform:
            # print(file_name)
            # print(image.shape[0])
            for i in range(image.shape[0]):
                oneimage = image[i]
                augmented = self.transform(image=oneimage)
                oneimage = augmented['image']
                augmented_images.append(oneimage)
        # label = torch.tensor(self.labels[idx]).long()
        return np.stack(augmented_images, axis=0)  # , label


import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.augmentations.transforms.JpegCompression(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


import torch.nn as nn
import timm


class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x

# state里面是pth文件路径
def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    count = 0
    for i, (images) in tk0:
        images = images[0]
        #print(images)
        images = images.to(device)
        avg_preds = []
        #print(i)
        # for 实际无意义
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
                # print(y_preds)
                # exit(0)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
        count += 1
        if count % 100 == 0:
            print(count)
    #probs = np.concatenate(probs)
    return np.asarray(probs)


if CFG.debug == True:  # 不调用
    short = short.sample(n=10)


# 这个文件夹里是训练好的 nocall detector
MODEL_DIR = '../input/clef-nocall-2class2-5fold/'
# 载入预训练模型
model = CustomResNext(CFG.model_name, pretrained=CFG.pretrained)
# 选一个模型
states = [torch.load(MODEL_DIR+f'{CFG.model_name}_fold{CFG.fold}_best.pth'), ]
# 获取数据 主要是根据路径载入全部文件 augmentation也做过了
test_dataset = TestDataset(short, transform=get_transforms(data='valid'))
# torch.utils.data
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True)

predictions = inference(model, states, test_loader, CFG.device)


predictions = [i[:,1] for i in predictions]
predictions = [' '.join(map(str, j.tolist())) for j in predictions]
short['nocalldetection'] = predictions
short.to_csv(f'./nocalldetection_for_shortaudio_fold{CFG.fold}.csv', index=False)



