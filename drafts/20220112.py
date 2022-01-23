import os
import wave
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def convert_file_to_wave(path):

    # 检查文件是否存在
    if not os.path.isfile(path):
        # 若不存在，则报错
        raise ValueError("文件不存在")
    else:
        # 若存在，则打开
        f = wave.open(path, 'rb')

    # 检查帧率是否为48000
    if f.getframerate() != 48000:
        raise ValueError("帧率必须是48000")

    # 检查是否为单声道
    if f.getnchannels() != 1:
        raise ValueError("必须是单声道")

    # 检查量化位数
    if f.getsampwidth() != 2:
        raise ValueError("量化位数必须是2")

    # 从文件读取波形
    buffer_signal = f.readframes(f.getnframes())  # 以二进制形式读取
    waveform = np.frombuffer(buffer_signal, np.short)  # 转换为帧，短整型
    if len(waveform) == 0:  # 检查波形长度
        raise ValueError("文件不能为空")
    rate = f.getframerate()  # 获取帧率
    f.close()   # 关闭

    # 对波形进行归一化处理
    signal = waveform / 32768.0  # 32768.0 float

    return rate, signal


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


def grayify_cmap(cmap):
    """Return a grayscale version of the colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def subplot_image(Sxx, n_subplot, title):
    cmap = grayify_cmap('cubehelix_r')
    # cmap = plt.cm.get_cmap('jet')
    # cmap = grayify_cmap('jet')
    plt.subplot(n_subplot)
    plt.title(title)
    plt.pcolormesh(Sxx, cmap=cmap)


def workflow_preprocess(path,

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

    # Read File
    rate, signal = convert_file_to_wave(path)

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

spec = workflow_preprocess(r"D:\birdsound\20211215\short(00-20-00-000)(00-20-05-000).wav")
print(spec.shape)