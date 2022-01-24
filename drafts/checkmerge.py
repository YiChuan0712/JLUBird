import wave as we
import numpy as np
import matplotlib.pyplot as plt

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

PATH = r'D:\birdsound'

f = we.open(PATH+r'\merge.wav', 'rb')

waveData, time = read_wav(f)

# f1 = we.open(PATH+r'\test211112.wav', 'rb')
#
# waveData, time = read_wav(f1)