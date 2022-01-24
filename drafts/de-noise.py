"""
此算法基于（但不是完全重现）Audacity概述的一种降噪效果的算法
该算法需要两个输入：

包含音频片段原型噪声的噪声音频片段
包含要删除的信号和噪声的信号音频片段
算法步骤

在噪声音频片段上计算FFT
统计信息是通过噪声的FFT计算得出的（频率）
基于噪声的统计信息（和算法的期望灵敏度）计算阈值
通过信号计算FFT
通过将信号FFT与阈值进行比较来确定掩码
使用滤镜在频率和时间上对蒙版进行平滑处理
掩码被叠加到信号的FFT中，并被反转
"""
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import wave as we
import time
from datetime import timedelta as td
import librosa
import scipy


def amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def db_to_amp(x, ):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def removeNoise(
        audio_clip,
        noise_clip,
        n_grad_freq=2,
        n_grad_time=4,
        n_fft=512,
        win_length=512,
        hop_length=256,
        n_std_thresh=1.5,
        prop_decrease=1.0,
):
    """Remove noise from audio based upon a clip containing only noise
    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
    Returns:
        array: The recovered signal with noise subtracted
    """

    # 在噪声上进行stft短时傅里叶变换
    noise_stft = librosa.stft(noise_clip, n_fft, hop_length, win_length)
    # 转dB分贝
    noise_stft_db = amp_to_db(np.abs(noise_stft))

    # 提取统计量 包括平均值 标准差
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    # 阈值设计为 平均值 + n_std_thresh倍标准差
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # 在音频上进行stft短时傅里叶变换
    sig_stft = librosa.stft(audio_clip, n_fft, hop_length, win_length)
    # 转dB分贝
    sig_stft_db = amp_to_db(np.abs(sig_stft))

    # 计算mask
    mask_gain_dB = np.min(amp_to_db(np.abs(sig_stft)))

    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T

    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh

    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # mask the signal
    sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )

    # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
            1j * sig_imag_masked
    )

    # recover the signal
    recovered_signal = librosa.istft(sig_stft_amp, hop_length, win_length)

    return recovered_signal




wav_loc = r"D:\birdsound\short(00-20-00-000)(00-30-00-000).wav"
src_rate, src_data = wavfile.read(wav_loc)
src_data = src_data.astype('float')


wav_loc = r"D:\birdsound\test211112-000100-000110.wav"
noise_rate, noise_data = wavfile.read(wav_loc)
noise_data = noise_data.astype('float')

noise_clip = noise_data

audio_clip_cafe = src_data

output = removeNoise(
    audio_clip=audio_clip_cafe,
    noise_clip=noise_clip,
    n_std_thresh=1,
    prop_decrease=0.95
)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
plt.plot(output, color="black")
ax.set_xlim((0, len(output)))
plt.show()

# 保存
save_wav = output.real.reshape((len(output), 1)).T.astype(np.short)
f = we.open(r"D:\birdsound\1234.wav", "wb")
# 配置声道数、量化位数和取样频率
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(48000)
# 将wav_data转换为二进制数据写入文件
f.writeframes(save_wav.tostring())
f.close()