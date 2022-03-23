"""
创建鸟声检测器，对7s的mel图进行检测，若检测到有鸟声则返回1，无则返回0

输入
    ../input/ff1010bird-duration7/        - 外部数据集，专门用于训练鸟声检测器，已经预先切割为7s并转为mel图像
    包含
        bird/        - 含鸟声
        nocall/        - 无鸟声
        rich_metadata.csv        - 元数据

输出
    ./oof_df.csv        - 每次训练模型时都是进行了5-fold划分，将占1/5的验证集部分和验证结果输出到oof_df.csv
    ./resnext50_32x4d_fold{n}_best.pth        - 训练好的检测器，{n}是验证fold
    ./train.log        - log文件

    将输出的检测器整理到 ../input/clef-nocall-2class2-5fold/
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
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from functools import partial
import cv2 # opencv_python
from PIL import Image
# (venv) D:\birdclefFirstPlace>pip install --upgrade pip setuptools wheel
# ERROR: To modify pip, please run the following command:
# D:\birdclefFirstPlace\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
# You are using pip version 10.0.1, however version 22.0.3 is available.
# You should consider upgrading via the 'python -m pip install --upgrade pip' command.
import torch  # 若出现问题 typing-extensions 退版本
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A  # pip install -U albumentations --no-binary qudida,albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import timm
import warnings
from sklearn.model_selection import StratifiedKFold
import pandas as pd

warnings.filterwarnings('ignore')
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

"""
配置CFG
"""
class CFG:
    print_freq = 100  # 打印数据间隔
    num_workers = 4  # load data 时使用
    model_name = 'resnext50_32x4d'
    dim = (128, 281)  # 图片尺寸
    scheduler = 'CosineAnnealingWarmRestarts'
    epochs = 10  # 训练轮数
    lr = 1e-4
    T_0 = 10  # for CosineAnnealingWarmRestarts
    min_lr = 5e-7  # for CosineAnnealingWarmRestarts
    batch_size = 32
    weight_decay = 1e-6
    max_grad_norm = 1000
    seed = 42
    target_size = 2
    target_col = 'hasbird'
    n_fold = 5
    pretrained = True
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

"""
meta数据读取
记录文件路径
train val划分
"""
# 读取rich_metadata的csv
# 	itemid	hasbird	filename	primary_label	filepath	frames	sr	duration	fold
train = pd.read_csv('../input/ff1010bird-duration7/rich_metadata.csv')
# 因为有无鸟叫的文件存在两个文件夹里，因此要分别重新记录路径
train.loc[train['hasbird'] == 0, 'filepath'] = '../input/ff1010bird-duration7/nocall/' + train.query('hasbird==0')['filename'] + '.npy'
train.loc[train['hasbird'] == 1, 'filepath'] = '../input/ff1010bird-duration7/bird/' + train.query('hasbird==1')['filename'] + '.npy'
# 丢弃含有空值的行 drop=True 就是把原来的索引index列去掉，重置index
train = train.dropna().reset_index(drop=True)

folds = train.copy()  # 复制一份csv
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)  # 五折划分 val占1/5 根据val给整个数据集分为五份
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[CFG.target_col])):
    # 因为val占1/5 每次给这1/5的标记对应的fold即可
    folds.loc[val_index, 'fold'] = int(n)  # 并且标明fold number
folds['fold'] = folds['fold'].astype(int)

"""
score & confusion matrix
"""
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

"""
logger
"""
@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    # 创建日志器对象
    logger = getLogger(__name__)
    # 设置日志最低输出级别为info
    logger.setLevel(INFO)
    # 再创建一个handler，用于输出到控制台
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    # 创建一个handler 用于写入日志文件
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = init_logger()

"""
random seed
"""
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

"""
数据集类
"""
class TrainDataset(Dataset):
    # df应该填入training set或test set
    def __init__(self, df, transform=None):
        self.df = df
        # 把filepath hasbird取出来
        self.file_paths = df['filepath'].values
        self.labels = df['hasbird'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_paths[idx]
        file_path = file_name
        image = np.load(file_path)
        image = image.transpose(1, 2, 0)
        image = np.squeeze(image)
        image = np.stack((image,) * 3, -1)  # 在最后一维stack
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        # 返回图片和label 即hasbird标签
        # image 是根据文件路径 从文件夹里直接读取的
        return image, label


"""
数据扩增手段
"""
def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.dim[0], CFG.dim[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.augmentations.transforms.JpegCompression(p=0.5),
            A.augmentations.transforms.ImageCompression(p=0.5,
                                                        compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP),
            # 使用Imagenet的均值和标准差是一种常见的做法
            # rgb
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),  # [h, w, c] to [c, h, w]
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


"""
神经网络
"""
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        # 全连接层
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x

# 保存并记录数据 类
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # value
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""
时间换算 进度计算
"""
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

"""
训练过程
"""

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()  # 启用batch normalization和drop out
    start = end = time.time()
    # global_step = 0

    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        # print(labels.size()) 32
        # print(batch_size)
        # exit(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)

        # record loss
        losses.update(loss.item(), batch_size)
        loss.backward()  # 将损失loss 向输入侧进行反向传播
        # 对于梯度爆炸问题，解决方法之一便是进行梯度剪裁
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        optimizer.step()  # optimizer.step()是优化器对x的值进行更新
        optimizer.zero_grad()  # 清空过往梯度

        # global_step += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                .format(
                epoch + 1, step + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(step + 1) / len(train_loader)),
                grad_norm=grad_norm,
            ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()  # 不启用 Batch Normalization 和 Dropout
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(
                step + 1, len(valid_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(step + 1) / len(valid_loader)),
            ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# def inference(model, states, test_loader, device):
#     model.to(device)
#     tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
#     probs = []
#     for i, (images) in tk0:
#         images = images.to(device)
#         avg_preds = []
#         for state in states:
#             model.load_state_dict(state['model'])
#             model.eval()
#             with torch.no_grad():
#                 y_preds = model(images)
#             avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
#         avg_preds = np.mean(avg_preds, axis=0)
#         probs.append(avg_preds)
#     probs = np.concatenate(probs)
#     return probs

""""""
""""""

def train_loop(train_folds, valid_folds):
    LOGGER.info(f"========== training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_dataset = TrainDataset(train_folds,
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds,
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    def get_scheduler(optimizer):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomResNext(CFG.model_name, pretrained=True)
    model.to(CFG.device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.
    best_loss = np.inf

    scores = []

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, CFG.device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, CFG.device)
        valid_labels = valid_folds[CFG.target_col].values

        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Accuracy: {score}')

        scores.append(score)

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'preds': preds},
                       OUTPUT_DIR + f'{CFG.model_name}_best.pth')

    check_point = torch.load(OUTPUT_DIR + f'{CFG.model_name}_best.pth')
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds, scores


"""
main
"""
def main(fold):
    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')  # <左对齐

    def get_result2(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        matrix = get_confusion_matrix(labels, preds)
        print('TN', matrix[0, 0])
        print('FP', matrix[0, 1])
        print('FN', matrix[1, 0])
        print('TP', matrix[1, 1])

    # train
    train_folds = folds.query(f'fold!={fold}').reset_index(drop=True)
    valid_folds = folds.query(f'fold=={fold}').reset_index(drop=False)
    oof_df, scores = train_loop(train_folds, valid_folds)
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    get_result2(oof_df)
    # save result
    oof_df.to_csv(OUTPUT_DIR + 'oof_df.csv', index=False)
    plt.plot([i for i in range(CFG.epochs)], scores)
    plt.title('valid score')
    plt.show()


if __name__ == '__main__':
    main(0)

