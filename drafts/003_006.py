# # -*- coding: utf-8 -*-
# import wave
# import pylab as pl
# import numpy as np
# # 打开WAV文档
# f = wave.open(r"D:\birdsound\short(00-20-00-000)(00-20-20-000).wav", "rb")
# # 读取格式信息
# # (nchannels, sampwidth, framerate, nframes, comptype, compname)
# params = f.getparams()
# nchannels, sampwidth, framerate, nframes = params[:4]
# print(nchannels, sampwidth, framerate, nframes)
# # 读取波形数据
# str_data = f.readframes(nframes)
# f.close()
# #将波形数据转换为数组
# wave_data = np.fromstring(str_data, dtype=np.short)
# wave_data.shape = -1, 2
# wave_data = wave_data.T
# time = np.arange(0, nframes) * (1.0 / framerate)
# # 绘制波形
# pl.subplot(211)
# pl.plot(time, wave_data[0])
# pl.subplot(212)
# pl.plot(time, wave_data[1], c="g")
# pl.xlabel("time (seconds)")
# pl.show()






from scipy.io import wavfile
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fs = 1000
#采样点数
num_fft = 512

"""
生成原始信号序列

在原始信号中加上噪声
np.random.randn(t.size)
"""
t = np.arange(0, 1, 1/fs)
f0 = 100
f1 = 200
x = np.cos(2*np.pi*f0*t) + 3*np.cos(2*np.pi*f1*t) + np.random.randn(t.size)

plt.figure(figsize=(15, 12))
ax=plt.subplot(511)
ax.set_title('original signal')
plt.tight_layout()
plt.plot(x)

"""
FFT(Fast Fourier Transformation)快速傅里叶变换
"""
Y = fft(x, num_fft)
Y = np.abs(Y)

ax=plt.subplot(512)
ax.set_title('fft transform')
plt.plot(20*np.log10(Y[:num_fft//2]))

"""
功率谱 power spectrum
直接平方
"""
ps = Y**2 / num_fft
ax=plt.subplot(513)
ax.set_title('direct method')
plt.plot(20*np.log10(ps[:num_fft//2]))

plt.show()





# import librosa
# import numpy as np
#
# n_fft = 512
# hop_length = 256
# win_length = 512
#
# src = r"D:\birdsound\short(00-20-00-000)(00-20-20-000).wav"
# wav, sample_rate = librosa.load(src, sr=48000, mono=True)
#
# y = np.pad(wav, int(n_fft // 2), mode='reflect')
# print(np.shape(y))
# y_frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length) # hop_length为帧移，librosa中默认取窗长的四分之一
# fft_window = librosa.filters.get_window('hann', win_length, fftbins=True)         # 窗长一般等于傅里叶变换维度，短则填充长则截断
# frames *= 0.5 - 0.5 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1)) # 原信号乘以汉宁窗函数






# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
#
# y, sr = librosa.load(r"D:\birdsound\short(00-20-00-000)(00-20-05-000).wav")
#
# #after(00-20-00-000)(00-20-05-000)
# #kaggle1(00-00-00-000)(00-00-20-000)
# C = np.abs(librosa.cqt(y, sr=sr))
#
# V = np.abs(librosa.vqt(y, sr=sr))
#
# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
#
# librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
#                          sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[0])
# ax[0].set(title='Constant-Q power spectrum', xlabel=None)
#
# ax[0].label_outer()
#
# img = librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
#                                sr=sr, x_axis='time', y_axis='cqt_note', ax=ax[1])
# ax[1].set_title('Variable-Q power spectrum')
#
# fig.colorbar(img, ax=ax, format="%+2.0f dB")
#
# plt.show()
#
# print(np.shape(librosa.amplitude_to_db(C, ref=np.max)))
