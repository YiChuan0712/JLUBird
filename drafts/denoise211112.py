"""
denoise211112.py by Yichuan
本文件是2021/11/12 进行的一些尝试去噪
频率域滤波 没感受到什么明显的效果

"""

import wave as we
import numpy as np
import matplotlib.pyplot as plt

PATH = r'D:\birdsound'

# 思路一
# 做傅里叶变换 在频率域进行滤波 然后再逆傅里叶变换
# 查资料发现 鸟的发声范围 为 2000 - 13000 Hz
# 因此 尝试只保留这个频率域的声音

def read_wav(wavfile, plots=True, normal=False):
    f = wavfile
    params = f.getparams()
    # print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.frombuffer(strData, dtype=np.int16)  # 将字符串转化为int
    # wave幅值归一化
    if normal == True:
        waveData = waveData*1.0/(max(abs(waveData)))
    # 绘图
    if plots == True:
        time = np.arange(0, nframes)*(1.0 / framerate)
        plt.figure(dpi=100)
        plt.plot(time, waveData)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Single channel wavedata")
        plt.show()
    return (waveData, time)

def fft_wav(waveData, plots=True):
    f_array = np.fft.fft(waveData)  # 傅里叶变换，结果为复数数组
    f_abs = f_array
    return f_abs

# """
f = we.open(PATH+r'\test211112-002500-002600.wav', 'rb')
waveData, time = read_wav(f)

wavefft = fft_wav(waveData)

# 截取
step_hz = 20000 / (len(waveData) / 2)
tab1_hz = 2000
tab_hz = 13000
savewav = []
for i in range(int(tab1_hz/step_hz)):
    savewav.append(0)
for i in range(int(tab1_hz/step_hz), int(tab_hz/step_hz)):
    savewav.append(wavefft[i])
for j in range(int(tab_hz/step_hz), (len(wavefft) - int(tab_hz/step_hz))):
    savewav.append(0)
for i in range((len(wavefft) - int(tab_hz/step_hz)), len(wavefft)):
    savewav.append(wavefft[i])


# 傅里叶逆变换
i_array = np.fft.ifft(savewav)


# 保存
save_wav = i_array.real.reshape((len(i_array), 1)).T.astype(np.short)
# print(save_wav.shape)
# i_array.real.tofile(PATH+r'\test.bin')

f = we.open(PATH+r'\test_denoising.wav', "wb")
# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(48000)
# 将wav_data转换为二进制数据写入文件
f.writeframes(save_wav.tostring())
f.close()
#"""

# 思路二
# 因为思路一效果不明显
# 继续查资料 认为可能是背景噪声太强导致的
# https://blog.csdn.net/bibinGee/article/details/105104158
# 可能这个方法可行
# 回头试试

#
