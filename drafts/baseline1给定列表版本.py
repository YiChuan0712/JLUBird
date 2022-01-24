"""
基线 - 第一步
音频转为图像（语谱图）
"""
import os
import warnings
import pandas as pd
import librosa
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
# 忽略所有warning
warnings.filterwarnings(action='ignore')

# 全局变量
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # 单位 秒
SPEC_SHAPE = (224, 224)  # 高 宽
FMIN = 20
FMAX = 16000

# 本基线来源于:
# https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-efficientnet


# 音频分割函数
# 提取语谱图并保存到指定路径
def get_spectrograms(filepath, primary_label, output_dir):
    # 用librosa打开文件 (仅处理前15s)
    sig, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None, duration=15)

    # 将音频按照5s的长度分割
    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # 音频结束?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)

    # 提取mel spectrogram
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:

        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                  sr=SAMPLE_RATE,
                                                  n_fft=2048,
                                                  hop_length=hop_length,
                                                  n_mels=SPEC_SHAPE[0],
                                                  fmin=FMIN,
                                                  fmax=FMAX)

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # 标准化
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()

        # 保存为图像
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] +
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")  # L 代表灰度图
        im.save(save_path)

        saved_samples.append(save_path)
        s_cnt += 1

    return saved_samples

thelist = ['amecro', 'cangoo', 'gockin', 'amegfi', 'amerob', 'rewbla', 'sonspa',
 'baloribalori', 'bkcchi', 'balori', 'comgra', 'grycat', 'haiwoo', 'norcarbalori',
 'rebwoo', 'belkin1', 'rewblabkcchi', 'eawpew', 'reevir1', 'reevir1bkcchi', 'norcar',
 'blujay', 'swaspablujay', 'swaspa', 'bobfly1', 'orfpar', 'plawre1', 'rucwar',
 'brnjaybrnjay', 'sthwoo1', 'bucmot2', 'rubwre1', 'rewblacangoo', 'chswarchswar',
 'chswar', 'ovenbi1clcrob', 'clcrob', 'comyel', 'eastow', 'crfpar', 'runwre1dowwoo',
 'gockingockin', 'rewblagockin', 'grekis', 'runwre1', 'grhcha1', 'norfli', 'hofwoo1',
 'whcpar', 'melbla1norcar', 'norwat', 'obnthr1', 'orcpar', 'ovenbi1', 'sonsparewbla',
 'rtlhum', 'yehcar1', 'woothr', 'yebsap']


# 读取metadata 格式dataframe
train = pd.read_csv(r'D:\birdclef-2021\train_metadata.csv', )

birds_count = {}  # 字典
for bird_species, count in zip(train.primary_label.unique(),  # zip用于打包成元组
                               train.groupby('primary_label')['primary_label'].count().values):
    if bird_species in thelist:
        birds_count[bird_species] = count  # 鸟种 - 音频条数
most_represented_birds = [key for key, value in birds_count.items()]

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())

print('训练集中的鸟种数量:', len(LABELS))
print('训练集中的音频文件数量:', len(TRAIN))
print('鸟种列表:', most_represented_birds)

TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)

print('最终保留的音频文件数量:', len(TRAIN))

input_dir = r'D:/birdclef-2021/train_short_audio/'
output_dir = r'D:/北美活跃/'
samples = []
with tqdm(total=len(TRAIN)) as pbar:
    for idx, row in TRAIN.iterrows():
        pbar.update(1)

        if row.primary_label in most_represented_birds:
            audio_file_path = os.path.join(input_dir, row.primary_label, row.filename)
            samples += get_spectrograms(audio_file_path, row.primary_label, output_dir)
print(samples)
str_samples = ','.join(samples)
TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
filename = open('im.txt', 'w')
filename.write(str_samples)
filename.close()


