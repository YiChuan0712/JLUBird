from scipy.io import wavfile  # scipy 1.7.3
import noisereduce as nr  # noisereduce 2.0.0
import numpy as np  # numpy 1.20.3
import librosa
import matplotlib.pyplot as plt

"""全局控制变量"""
SAMPLE_RATE = 48000

def denoise_by_noisereduce(filepath,
                           savepath,
                           stationary=False,
                           prop_decrease=1.0,
                           time_constant_s=2.0,
                           freq_mask_smooth_hz=500,
                           time_mask_smooth_ms=50,
                           thresh_n_mult_nonstationary=1,
                           sigmoid_slope_nonstationary=10,
                           n_std_thresh_stationary=1.5,
                           tmp_folder=None,
                           chunk_size=60000,
                           n_fft=1024,
                           win_length=None,
                           hop_length=None,
                           n_jobs=1
                           ):
    """
    去噪方法1 - 主要利用 noisereduce.reduce_noise 工具
    https://pypi.org/project/noisereduce/

    :param filepath : 输入文件路径
    :param savepath : 保存文件路径
    :param stationary : 设置是否使用stationary去噪, 默认输入False
    :param prop_decrease : 去噪比例, 默认输入1.0, 代表去除100%的噪声
    :param time_constant_s : 时间常数, 用于在non-stationary去噪中计算地板噪声, 默认输入2.0
    :param freq_mask_smooth_hz : 频率范围, 用于对掩码进行平滑, 单位Hz, 默认输入500
    :param time_mask_smooth_ms : 时间范围, 用于对掩码进行平滑, 单位ms, 默认输入50
    :param thresh_n_mult_nonstationary : 只在non-stationary中使用, 默认输入1
    :param sigmoid_slope_nonstationary : 只在non-stationary中使用, 默认输入10
    :param n_std_thresh_stationary : 标准差倍数n, 平均之上n倍标准差的位置用于放置阈值, 以区分信号和噪音, 默认输入1.5
    :param tmp_folder : 临时folder, 用于写入并行处理时的波形, 默认是python的default temp folder, 默认输入None
    :param chunk_size : 用于去噪的signal chunk大小, 大值占用内存, 小值占用计算时间, 默认输入60000
    :param n_fft : 快速傅里叶变换长度, 默认输入1024
    :param win_length : 窗长, 若无特殊要求就与n_fft相同, 默认输入None
    :param hop_length : 重叠, 若无特殊要求就与win_length // 4相同, 默认输入None
    :param n_jobs : 并行线程数, 设为-1即可使用所有CPU cores, 默认输入1

    :return "SUCCESS" : 去噪成功
    :return 其他 : 去噪失败
    """

    # 读取
    signal, rate = librosa.load(filepath, sr=SAMPLE_RATE, offset=None)

    # 去噪
    reduced_noise = nr.reduce_noise(y=signal,
                                    sr=rate,
                                    stationary=stationary,
                                    prop_decrease=prop_decrease,
                                    time_constant_s=time_constant_s,
                                    freq_mask_smooth_hz=freq_mask_smooth_hz,
                                    time_mask_smooth_ms=time_mask_smooth_ms,
                                    thresh_n_mult_nonstationary=thresh_n_mult_nonstationary,
                                    sigmoid_slope_nonstationary=sigmoid_slope_nonstationary,
                                    n_std_thresh_stationary=n_std_thresh_stationary,
                                    tmp_folder=tmp_folder,
                                    chunk_size=chunk_size,
                                    n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    n_jobs=n_jobs
                                    )

    # 写入
    wavfile.write(savepath, rate, reduced_noise)

    return "SUCCESS"

def preprocess_sound_file(filename, class_dir, noise_dir, segment_size_seconds):
    """ Preprocess sound file. Loads sound file from filename, downsampels,
    extracts signal/noise parts from sound file, splits the signal/noise parts
    into equally length segments of size segment size seconds.
    # Arguments
        filename : the sound file to preprocess
        class_dir : the directory to save the extracted signal segments in
        noise_dir : the directory to save the extracted noise segments in
        segment_size_seconds : the size of each segment in seconds
    # Returns
        nothing, simply saves the preprocessed sound segments
    """

    samplerate, wave = read_wave_file_not_normalized(filename)

    # 如果是空文件
    if len(wave) == 0:
        print("An empty sound file..")
        wave = np.zeros(samplerate * segment_size_seconds, dtype=np.int16)

    #
    signal_wave, noise_wave = preprocess_wave(wave, samplerate)

    print(signal_wave.shape)
    print(noise_wave.shape)

    # 如果矩阵的行数为0
    if signal_wave.shape[0] == 0:
        signal_wave = np.zeros(samplerate * segment_size_seconds, dtype=np.int16)

    basename = get_basename_without_ext(filename)

    # print(filename)
    # print(class_dir)
    # print(noise_dir)

    if signal_wave.shape[0] > 0:
        signal_segments = split_into_segments(signal_wave, samplerate, segment_size_seconds)
        save_segments_to_file(class_dir, signal_segments, basename, samplerate)
    if noise_wave.shape[0] > 0:
        noise_segments = split_into_segments(noise_wave, samplerate, segment_size_seconds)
        save_segments_to_file(noise_dir, noise_segments, basename, samplerate)

# # load data
filepath = r"D:\birdsound\20211122\origin.wav"
savepath = r"D:\birdsound\20211122\test0124-80.wav"
denoise_by_noisereduce(filepath, savepath, prop_decrease=0.8, n_std_thresh_stationary=1.1)




