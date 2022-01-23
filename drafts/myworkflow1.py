"""
处理第一步
音频转为图像（语谱图）
需要加入显示图像大小的功能
"""
import os
import warnings
import pandas as pd
import librosa
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from tqdm import tqdm
import scipy.signal
import random
# 忽略所有warning
warnings.filterwarnings(action='ignore')

# 全局变量
RANDOM_SEED = 42
RANDOM = np.random.RandomState(RANDOM_SEED)
SIGNAL_LENGTH = 147584
NOISE_PATH = ""


def time_warp(spec, max_time_warp=16):
    """time warp for spec augment
    move random center frame by the random width ~ uniform(-window, window)
    :param numpy.ndarray x: spectrogram (time, freq)
    :param int max_time_warp: maximum time frames to warp
    :param bool inplace: overwrite x with the result
    :param str mode: "PIL" (default, fast, not differentiable) or "sparse_image_warp"
        (slow, differentiable)
    :returns numpy.ndarray: time warped spectrogram (time, freq)
    paddle speech
    PIL模式
    """
    window = max_time_warp

    t = spec.shape[0]

    if t - window <= window:

        return spec

    # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
    center = random.randrange(window, t - window)
    warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1

    from PIL.Image import BICUBIC

    left = Image.fromarray(spec[:center]).resize((spec.shape[1], warped), BICUBIC)
    right = Image.fromarray(spec[center:]).resize((spec.shape[1], t - warped),
                                                   BICUBIC)

    return np.concatenate((left, right), 0)


def augment(spec):

    # Parse selected augmentations
    for aug in ['h_roll', 'warp', 'noise']:

        # Decide if we should apply this method
        if RANDOM.choice([True, False], p=[0.5, 0.5]):

            # Apply augmentation
            if aug == 'h_roll':
                # Vertical Roll
                vertical = 0.1
                spec = np.roll(spec, int(spec.shape[0] * RANDOM.uniform(-vertical, vertical)), axis=0)

            if aug == 'warp':
                spec = time_warp(spec)

            if aug == 'noise':
                amount = 0.05
                spec += RANDOM.normal(0.0, RANDOM.uniform(0, amount ** 0.5), spec.shape)
                spec = np.clip(spec, 0.0, 1.0)

    return spec


def apply_bandpass_filter(signal, rate, fmin=150, fmax=15000, N=4):

    # 将频率范围转为指定的格式
    # Wn是归一化截止频率
    # Wn = 2 * 截止频率 / 采样频率
    # 根据采样定理，采样频率要大于两倍的信号本身最大的频率，才能还原信号
    # 截止频率一定小于信号本身最大的频率，所以Wn一定在0和1之间
    # 当构造带通滤波器时，Wn为长度为2的列表
    Wn = [fmin * 2.0 / float(rate), fmax * 2.0 / float(rate)]

    # 构造滤波器（带通滤波，返回IIR滤波器的二阶截面表示）
    filter = scipy.signal.butter(N, Wn, btype='bandpass', output='sos')

    # 过滤
    signal = scipy.signal.sosfiltfilt(filter, signal)

    return signal


def convert_wave_to_spec(signal, rate, wlen=512, woverlap=128, nfft=512):

    # 封装scipy.signal.spectrogram
    flist, tlist, spec = scipy.signal.spectrogram(signal,
                                          fs=rate,
                                          window=scipy.signal.windows.hann(wlen),
                                          nperseg=wlen,
                                          noverlap=woverlap,
                                          nfft=nfft,
                                          detrend=False,
                                          mode='magnitude')
    return flist, tlist, spec


def apply_power_spec(spec):

    # 将语谱图转为能量谱
    spec = spec ** 2
    return spec


def apply_log_spec(spec, m=100):

    # 将语谱图转为log谱（分贝）
    r = np.max(spec)
    # 阈值
    # m = 100
    # 先转为能量谱
    spec = spec ** 2
    # 10lg(spec/max)
    # 如果 spec < 1e-10， 取1e-10
    spec = 10.0 * np.log10(np.maximum(1e-10, spec) / r)
    spec = np.maximum(spec, spec.max() - m)
    return spec


def apply_nonlinear_spec(spec, a=-1.2):

    # 非线性方法 by Schluter, 2018
    # a取[-1.7, -1.2]，取值越大，对噪声抑制越强
    # a = -1.2  # Higher values yield better noise suppression
    sigma = 1.0 / (1.0 + np.exp(-a))
    spec = spec ** sigma
    return spec


def apply_PCEN_spec(spec, rate, woverlap=128, gain=0.8, bias=10, power=0.25, t=0.060, eps=1e-6):

    # Per-Channel Energy Normalization
    # Trainable Frontend For Robust and Far-Field Keyword Spotting
    # Per-Channel Energy Normalization: Why and How
    # gain = 0.8
    # bias = 10
    # power = 0.25
    # t = 0.060
    # eps = 1e-6

    s = 1 - np.exp(- float(woverlap) / (t * rate))
    M = scipy.signal.lfilter([s], [1, s - 1], spec)
    smooth = (eps + M) ** (-gain)
    spec = (spec * smooth + bias) ** power - bias ** power
    return spec


def apply_BirdNet_MFCC(spec, flist, fmin=150, fmax=15000, banknum=64, A=4581.0, fbreak=1750.0):

    # 元素应该插入在数组中那个位置上
    begin = flist.searchsorted(fmin, side='left')
    end = flist.searchsorted(fmax, side='right')

    # BirdNet P69
    # A = 4581.0
    # fbreak = 1750.0

    # Hz转mel
    frange = A * np.log10(1 + np.asarray([fmin, fmax]) / fbreak)
    # 获取等距的mel采样点
    melpoints = np.linspace(frange[0], frange[1], banknum + 2)
    # mel转Hz
    bankpoints = (fbreak * (10 ** (melpoints / A) - 1))

    filterbank = np.zeros([len(flist), banknum])
    for i in range(1, banknum + 1):
        # 把flist中在[bankpoints[i - 1], bankpoints[i]]之间的标成True，对应频率（spectrogram中的行）
        mask = np.logical_and(flist >= bankpoints[i - 1], flist <= bankpoints[i])
        filterbank[mask, i - 1] = (flist[mask] - bankpoints[i - 1]) / (bankpoints[i] - bankpoints[i - 1])
        # 把flist中在[bankpoints[i], bankpoints[i + 1]]之间的标成True，对应频率（spectrogram中的行）
        mask = np.logical_and(flist >= bankpoints[i], flist <= bankpoints[i + 1])
        filterbank[mask, i - 1] = (bankpoints[i + 1] - flist[mask]) / (bankpoints[i + 1] - bankpoints[i])

    # 此时filterbank中的[x,y]，如果第x个频率在y计算出的bankpoints范围内，就会有值，否则保持为0
    # flist的多个值可以属于同一个bank范围
    # normalization
    temp = filterbank.sum(axis=0)
    nzmask = temp != 0
    filterbank[:, nzmask] /= np.expand_dims(temp[nzmask], 0)

    # clip
    filterbank = filterbank[begin:end, :]

    # clip spec
    # 要转置
    spec = np.transpose(spec[begin:end, :], [1, 0])  # y, x
    spec = np.dot(spec, filterbank)

    spec = np.transpose(spec, [1, 0])

    return spec


def workflow_preprocess(signal,
                        rate,

                        trim0=64,
                        trim1=384,

                        bp=True,
                        fmin=150,
                        fmax=15000,
                        N=4,

                        wlen=512,
                        woverlap=128,
                        nfft=512,

                        mfcc=True,
                        banknum=64,
                        A=4581.0,
                        fbreak=1750.0,

                        magmode='nonlinear',

                        m=100,

                        a=-1.2,

                        gain=0.8,
                        bias=10,
                        power=0.25,
                        t=0.060,
                        eps=1e-6
                        ):

    # Bandpass Filter
    if bp:
        signal = apply_bandpass_filter(signal, rate, fmin=fmin, fmax=fmax, N=N)

    # Spectrogram
    flist, tlist, spec = convert_wave_to_spec(signal, rate, wlen=wlen, woverlap=woverlap, nfft=nfft)

    # MFCC
    if mfcc:
        spec = apply_BirdNet_MFCC(spec, flist, fmin=fmin, fmax=fmax, banknum=banknum, A=A, fbreak=fbreak)

    # Magnitude transformation
    if magmode == 'pass':
        pass
    elif magmode == 'power':
        spec = apply_power_spec(spec)
    elif magmode == 'log':
        spec = apply_log_spec(spec, m=m)
    elif magmode == 'nonlinear':
        spec = apply_nonlinear_spec(spec, a=a)
    elif magmode == 'pcen':
        spec = apply_PCEN_spec(spec, rate, woverlap=woverlap, gain=gain, bias=bias, power=power, t=t, eps=eps)
    else:
        raise ValueError("magnitude模式选择错误")

    # Flip
    spec = spec[::-1, ...]

    # Trim
    spec = spec[:trim0, :trim1]

    # Normalization
    spec -= spec.min()
    spec /= spec.max()

    return spec


# 音频分割函数
# 提取语谱图并保存到指定路径
def get_spectrograms(filepath, primary_label, output_dir):
    # 用librosa打开文件
    signal, rate = librosa.load(filepath, sr=48000, offset=None)

    # 将音频按照大约3s的长度分割
    sig_splits = []
    for i in range(0, len(signal), int(SIGNAL_LENGTH)):
        split = signal[i:i + int(SIGNAL_LENGTH)]

        # 音频结束?
        if len(split) < int(SIGNAL_LENGTH):
            break

        sig_splits.append(split)

    # 提取mel spectrogram
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:

        spec = workflow_preprocess(chunk, rate)
        spec = augment(spec)

        # 保存为图像
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] +
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(spec * 255.0).convert("L")  # L 代表灰度图
        im.save(save_path)

        saved_samples.append(save_path)
        s_cnt += 1

    return saved_samples


# 读取metadata 格式dataframe
train = pd.read_csv(r'D:\birdclef-2021\train_metadata.csv', )

birds_count = {}  # 字典
for bird_species, count in zip(train.primary_label.unique(),  # zip用于打包成元组
                               train.groupby('primary_label')['primary_label'].count().values):
    birds_count[bird_species] = count  # 鸟种 - 音频条数
most_represented_birds = [key for key, value in birds_count.items()]

TRAIN = train.query('primary_label in @most_represented_birds')
LABELS = sorted(TRAIN.primary_label.unique())

print('训练集中的鸟类物种数:', len(LABELS))
print('训练集中的音频文件数:', len(TRAIN))
print('鸟类物种列表:', most_represented_birds)

TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)

#print('最终保留的音频文件数量:', len(TRAIN))

"""
input_dir = r'D:/birdclef-2021/train_short_audio/'
output_dir = r'D:/20220119完整预处理无数据增强/'
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
filename = open('im20220119.txt', 'w')
filename.write(str_samples)
filename.close()
# """


# """
# 测试用
input_dir = r'D:/test/in/'
output_dir = r'D:/test/out/'
samples = []
get_spectrograms('D:/test/in/XC135079.ogg', "", output_dir)
# """
