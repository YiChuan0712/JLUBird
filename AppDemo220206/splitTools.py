import numpy as np
import os
import wave
import librosa
from skimage import morphology
from scipy.io import wavfile
from matplotlib import pyplot as plt

"""
stft的输入wave需要是float格式 用.astype('float')转化即可
"""

def wave_to_amplitude_spectrogram(wave, fs):
    X = librosa.stft(wave.astype('float'), n_fft=512, hop_length=128, win_length=512)
    X = np.abs(X) ** 2
    #return X[4:232]
    return X

def wave_to_log_amplitude_spectrogram(wave, fs):
    return np.log(wave_to_amplitude_spectrogram(wave, fs))

def get_basename_without_ext(filepath):
    basename = os.path.basename(filepath).split(os.extsep)[0]
    return basename

def read_wave_file_not_normalized(filename):
    """
        读文件 返回wave和采样率
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")

    s = wave.open(filename, 'rb')

    if (s.getnchannels() != 1):
        raise ValueError("Wave file should be mono")
    # if (s.getframerate() != 22050):
        # raise ValueError("Sampling rate of wave file should be 16000")

    # 获取波形数据
    strsig = s.readframes(s.getnframes())

    # 将波形数据转换为数组
    x = np.fromstring(strsig, np.short)

    # 获取采样率
    fs = s.getframerate()
    s.close()

    return fs, x

def compute_signal_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
    # Returns
        binary_mask : the binary signal mask
    """
    threshold = 3
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    return mask

def compute_noise_mask(spectrogram):
    """ Computes a binary noise mask (convenience function)
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
    # Returns
        binary_mask : the binary noise mask
    """
    threshold = 2.5
    mask = compute_binary_mask_sprengel(spectrogram, threshold)
    # invert mask
    return np.logical_not(mask)

def normalize(X):
    """ Normalize numpy array to interval [0, 1]
    """
    mi = np.min(X)
    ma = np.max(X)

    X = (X-mi)/(ma-mi)
    return X

def median_clipping(spectrogram, number_times_larger):
    """ Compute binary image from spectrogram where cells are marked as 1 if
    number_times_larger than the row AND column median, otherwise 0
    """
    row_medians = np.median(spectrogram, axis=1)
    col_medians = np.median(spectrogram, axis=0)

    # create 2-d array where each cell contains row median
    # 平铺
    row_medians_cond = np.tile(row_medians, (spectrogram.shape[1], 1)).transpose()
    # create 2-d array where each cell contains column median
    col_medians_cond = np.tile(col_medians, (spectrogram.shape[0], 1))

    # find cells number_times_larger than row and column median
    larger_row_median = spectrogram >= row_medians_cond*number_times_larger
    larger_col_median = spectrogram >= col_medians_cond*number_times_larger

    # create binary image with cells number_times_larger row AND col median
    binary_image = np.logical_and(larger_row_median, larger_col_median)
    return binary_image

def smooth_mask(mask):
    """ Smooths a binary mask using 4x4 dilation
        # Arguments
            mask : the binary mask
        # Returns
            mask : a smoother binary mask
    """
    n_hood = np.ones(4)
    mask = morphology.binary_dilation(mask, n_hood)
    mask = morphology.binary_dilation(mask, n_hood)

    # type casting is a bitch
    return mask

def compute_binary_mask_sprengel(spectrogram, threshold):
    """ Computes a binary mask for the spectrogram
    # Arguments
        spectrogram : a numpy array representation of a spectrogram (2-dim)
        threshold   : a threshold for times larger than the median
    # Returns
        binary_mask : the binary mask
    """
    # normalize to [0, 1)
    norm_spectrogram = normalize(spectrogram)

    # median clipping
    binary_image = median_clipping(norm_spectrogram, threshold)

    # erosion
    binary_image = morphology.binary_erosion(binary_image, selem=np.ones((4, 4)))

    # dilation
    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    # extract mask
    mask = np.array([np.max(col) for col in binary_image.T])
    mask = smooth_mask(mask)

    return mask

# TODO: This method needs some real testing
def reshape_binary_mask(mask, size):
    """ Reshape a binary mask to a new larger size
    """
    reshaped_mask = np.zeros(size, dtype=bool)

    x_size_mask = mask.shape[0]
    scale_fact = int(np.floor(size/x_size_mask))
    rest_fact = float(size)/x_size_mask - scale_fact

    rest = rest_fact
    i_begin = 0
    i_end = int(scale_fact + np.floor(rest))
    for i in mask:
        reshaped_mask[i_begin:i_end] = i
        rest += rest_fact
        i_begin = i_end
        i_end = i_end + scale_fact + int(np.floor(rest))
        if rest >= 1:
            rest -= 1.

    if not (i_end - scale_fact) == size:
        raise ValueError("there seems to be a scaling error in reshape_binary_mask")

    return reshaped_mask

def extract_masked_part_from_wave(mask, wave):
    return wave[mask]

def preprocess_wave(wave, fs):
    """ Preprocess a signal by computing the noise and signal mask of the
    signal, and extracting each part from the signal
    """
    Sxx = wave_to_amplitude_spectrogram(wave, fs)

    n_mask = compute_noise_mask(Sxx)
    s_mask = compute_signal_mask(Sxx)

    n_mask_scaled = reshape_binary_mask(n_mask, wave.shape[0])
    s_mask_scaled = reshape_binary_mask(s_mask, wave.shape[0])

    signal_wave = extract_masked_part_from_wave(s_mask_scaled, wave)
    noise_wave = extract_masked_part_from_wave(n_mask_scaled, wave)

    return signal_wave, noise_wave

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

def save_segments_to_file(output_dir, segments, basename, samplerate):
    # print("save segments ({}) to file".format(str(len(segments))))
    i_segment = 0
    for segment in segments:
        segment_filepath = os.path.join(output_dir, basename + "_seg_" + str(i_segment) + ".wav")
        # print("save segment: {}".format(segment_filepath))
        write_wave_to_file(segment_filepath, samplerate, segment)
        i_segment += 1

def split_into_segments(wave, samplerate, segment_time):
    """ Split a wave into segments of segment_size. Repeat signal to get equal
    length segments.
    """
    # print("split into segments")
    segment_size = samplerate * segment_time
    wave_size = wave.shape[0]

    nb_repeat = segment_size - (wave_size % segment_size)
    nb_tiles = 2
    if wave_size < segment_size:
        nb_tiles = int(np.ceil(segment_size/wave_size))
    repeated_wave = np.tile(wave, nb_tiles)[:wave_size+nb_repeat]
    nb_segments = repeated_wave.shape[0]/segment_size

    if not repeated_wave.shape[0] % segment_size == 0:
        raise ValueError("reapeated wave not even multiple of segment size")

    segments = np.split(repeated_wave, int(nb_segments), axis=0)

    return segments

def write_wave_to_file(filename, rate, wave):
    wavfile.write(filename, rate, wave)

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

def read_wave_file(filename):
    """ Read a wave file from disk
    # Arguments
        filename : the name of the wave file
    # Returns
        (fs, x)  : (sampling frequency, signal)
    """
    if (not os.path.isfile(filename)):
        raise ValueError("File does not exist")

    s = wave.open(filename, 'rb')

    if (s.getnchannels() != 1):
        raise ValueError("Wave file should be mono")
    # if (s.getframerate() != 22050):
        # raise ValueError("Sampling rate of wave file should be 16000")

    strsig = s.readframes(s.getnframes())
    x = np.fromstring(strsig, np.short)
    fs = s.getframerate()
    s.close()

    x = x/32768.0

    return fs, x

def sprengel_binary_mask_from_wave_file(filepath):
    fs, x = read_wave_file(filepath)
    Sxx = wave_to_amplitude_spectrogram(x, fs)
    Sxx_log = wave_to_log_amplitude_spectrogram(x, fs)

    # plot spectrogram
    fig = plt.figure(1)
    subplot_image(Sxx_log, 411, "Spectrogram")

    Sxx = normalize(Sxx)
    binary_image = median_clipping(Sxx, 3.0)

    subplot_image(binary_image + 0, 412, "Median Clipping")

    binary_image = morphology.binary_erosion(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 413, "Erosion")

    binary_image = morphology.binary_dilation(binary_image, selem=np.ones((4, 4)))

    subplot_image(binary_image + 0, 414, "Dilation")

    mask = np.array([np.max(col) for col in binary_image.T])
    mask = morphology.binary_dilation(mask, np.ones(4))
    mask = morphology.binary_dilation(mask, np.ones(4))

    # plot_vector(mask, "Mask")

    fig.set_size_inches(10, 12)
    plt.tight_layout()
    fig.savefig(get_basename_without_ext(filepath) + "_binary_mask.png", dpi=100)


"""
另一种方法
"""

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import eyed3
import hmmlearn
import plotly
import tqdm
from pydub import AudioSegment
import imblearn


def split_by_pyAudioAnalysis(filepath, savepath, st_win=5, st_step=0.5, smoothWindow=0.9, weight=0.2):
    """
    分片方法1 - 主要利用 audioSegmentation.silence_removal 工具
    https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioSegmentation.py

    :param filepath : 输入文件路径
    :param savepath : 保存文件路径
    :param st_win : window size in seconds
    :param st_step : window step in seconds
    :param smoothWindow : (optinal) smooth window (in seconds)
    :param weight : (optinal) weight factor (0 < weight < 1)
                              the higher, the more strict

    :return 切片列表
    """

    [Fs, x] = aIO.read_audio_file(filepath)

    segments = aS.silence_removal(x, Fs, st_win, st_step, smoothWindow, weight, plot=False)
    print(segments)

    def ms_to_time(total_ms=0):
        total_sec = int(total_ms / 1000)
        millisecond = int(total_ms - total_sec * 1000)
        hour = int(total_sec / 3600)
        total_sec = total_sec - hour * 3600
        minute = int(total_sec / 60)
        second = int(total_sec - minute * 60)
        return hour, minute, second, millisecond

    # wav分割函数
    def split_save_wav(filepath, savepath, begin, end):

        sound = AudioSegment.from_wav(filepath)

        begin_tuple = ms_to_time(begin)
        end_tuple = ms_to_time(end)

        section1 = "(" + str(begin_tuple[0]).zfill(2) + "-" \
                   + str(begin_tuple[1]).zfill(2) + "-" \
                   + str(begin_tuple[2]).zfill(2) + "-" \
                   + str(begin_tuple[3]).zfill(3) \
                   + ")" \
                   + "(" + str(end_tuple[0]).zfill(2) + "-" \
                   + str(end_tuple[1]).zfill(2) + "-" \
                   + str(end_tuple[2]).zfill(2) + "-" \
                   + str(end_tuple[3]).zfill(3) \
                   + ")"
        savepath = savepath.rsplit('.', 1)[0] + section1 + '.wav'
        print(savepath)

        cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
        cut_wav.export(savepath, format='wav')  # 存储新的wav文件

        print("已导出到" + savepath)

    for i in segments:
        split_save_wav(filepath, savepath, int(i[0] * 1000), int(i[1] * 1000))

    return segments

# split_by_pyAudioAnalysis(r"D:\birdsound\merge2.wav",r"D:\birdsound\merge2.wav")

