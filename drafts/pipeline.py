"""
@software{tim_sainburg_2019_3243139,
  author       = {Tim Sainburg},
  title        = {timsainb/noisereduce: v1.0},
  month        = jun,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {db94fe2},
  doi          = {10.5281/zenodo.3243139},
  url          = {https://doi.org/10.5281/zenodo.3243139}
}


@article{sainburg2020finding,
  title={Finding, visualizing, and quantifying latent structure across diverse animal vocal repertoires},
  author={Sainburg, Tim and Thielk, Marvin and Gentner, Timothy Q},
  journal={PLoS computational biology},
  volume={16},
  number={10},
  pages={e1008228},
  year={2020},
  publisher={Public Library of Science}
}

y : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
      input signal
  sr : int
      sample rate of input signal / noise signal
  y_noise : np.ndarray [shape=(# frames,) or (# channels, # frames)], real-valued
      noise signal to compute statistics over (only for non-stationary noise reduction).
  stationary : bool, optional
      Whether to perform stationary, or non-stationary noise reduction, by default False
  prop_decrease : float, optional
      The proportion to reduce the noise by (1.0 = 100%), by default 1.0
  time_constant_s : float, optional
      The time constant, in seconds, to compute the noise floor in the non-stationary
      algorithm, by default 2.0
  freq_mask_smooth_hz : int, optional
      The frequency range to smooth the mask over in Hz, by default 500
  time_mask_smooth_ms : int, optional
      The time range to smooth the mask over in milliseconds, by default 50
  thresh_n_mult_nonstationary : int, optional
      Only used in nonstationary noise reduction., by default 1
  sigmoid_slope_nonstationary : int, optional
      Only used in nonstationary noise reduction., by default 10
  n_std_thresh_stationary : int, optional
      Number of standard deviations above mean to place the threshold between
      signal and noise., by default 1.5
  tmp_folder : [type], optional
      Temp folder to write waveform to during parallel processing. Defaults to
      default temp folder for python., by default None
  chunk_size : int, optional
      Size of signal chunks to reduce noise over. Larger sizes
      will take more space in memory, smaller sizes can take longer to compute.
      , by default 60000
      padding : int, optional
      How much to pad each chunk of signal by. Larger pads are
      needed for larger time constants., by default 30000
  n_fft : int, optional
      length of the windowed signal after padding with zeros.
      The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``.
      The default value, ``n_fft=2048`` samples, corresponds to a physical
      duration of 93 milliseconds at a sample rate of 22050 Hz, i.e. the
      default sample rate in librosa. This value is well adapted for music
      signals. However, in speech processing, the recommended value is 512,
      corresponding to 23 milliseconds at a sample rate of 22050 Hz.
      In any case, we recommend setting ``n_fft`` to a power of two for
      optimizing the speed of the fast Fourier transform (FFT) algorithm., by default 1024
  win_length : [type], optional
      Each frame of audio is windowed by ``window`` of length ``win_length``
      and then padded with zeros to match ``n_fft``.
      Smaller values improve the temporal resolution of the STFT (i.e. the
      ability to discriminate impulses that are closely spaced in time)
      at the expense of frequency resolution (i.e. the ability to discriminate
      pure tones that are closely spaced in frequency). This effect is known
      as the time-frequency localization trade-off and needs to be adjusted
      according to the properties of the input signal ``y``.
      If unspecified, defaults to ``win_length = n_fft``., by default None
  hop_length : [type], optional
      number of audio samples between adjacent STFT columns.
      Smaller values increase the number of columns in ``D`` without
      affecting the frequency resolution of the STFT.
      If unspecified, defaults to ``win_length // 4`` (see below)., by default None
  n_jobs : int, optional
      Number of parallel jobs to run. Set at -1 to use all CPU cores, by default 1

"""

from scipy.io import wavfile
import noisereduce as nr
import matplotlib.pyplot as plt

# load data
rate, data = wavfile.read(r"D:\birdsound\20211122\origin.wav")
# perform noise reduction
prop_decrease = 0.8
n_std_thresh_stationary = 1.1
reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=prop_decrease, n_std_thresh_stationary=n_std_thresh_stationary)
wavfile.write(r"D:\birdsound\20211122\after de-noise.wav", rate, reduced_noise)

rate, data = wavfile.read(r"D:\birdsound\20211122\origin.wav")
plt.subplot(211)
plt.plot(data)
plt.title('origin audio')
plt.tight_layout()

rate1, data1 = wavfile.read(r"D:\birdsound\20211122\after de-noise.wav")
plt.subplot(212)
plt.plot(data1)
plt.title('after de-noise')
plt.tight_layout()

plt.show()

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import eyed3
import hmmlearn
import plotly
import tqdm


[Fs, x] = aIO.read_audio_file(r"D:\birdsound\20211122\after de-noise.wav",)

segments = aS.silence_removal(x, Fs, 5, 0.5, 0.9, 0.2, plot = True)

from pydub import AudioSegment

#segments = [[281.0, 284.5], [329.0, 332.0], [343.5, 344.0], [356.5, 360.5], [408.0, 409.5], [411.5, 412.5], [415.0, 416.0], [427.0, 428.0], [434.5, 436.5], [441.5, 446.5], [448.5, 449.0], [450.5, 452.0], [457.5, 458.0], [460.5, 543.0], [545.0, 768.0], [772.0, 928.5], [930.0, 1154.5]]

def split_and_save(input_file, output_file, begin=0, end=0):

    sound = AudioSegment.from_wav(input_file)
    duration = sound.duration_seconds * 1000  # 音频时长 ms

    if 0 <= begin < end <= duration is False:
        return "make sure 0 <= begin < end <= duration"

    begin_tuple = ms_to_time(begin)
    end_tuple = ms_to_time(end)

    section1 = "(" + str(begin_tuple[0]).zfill(2) + "-"\
                   + str(begin_tuple[1]).zfill(2) + "-"\
                   + str(begin_tuple[2]).zfill(2) + "-"\
                   + str(begin_tuple[3]).zfill(3)\
                   + ")"\
                   + "(" + str(end_tuple[0]).zfill(2) + "-" \
                   + str(end_tuple[1]).zfill(2) + "-" \
                   + str(end_tuple[2]).zfill(2) + "-" \
                   + str(end_tuple[3]).zfill(3) \
                   + ")"
    output_file = output_file.rsplit('.',1)[0] + section1 + '.wav'

    cut_wav = sound[begin:end]  # 以毫秒为单位截取[begin, end]区间的音频
    cut_wav.export(output_file, format='wav')  # 存储新的wav文件

    print(output_file)

def ms_to_time(total_ms=0):

    if isinstance(total_ms, int) is False:
        return "'total_ms' should be of type 'int'"
    if total_ms < 0:
        return "'total_ms' should be in the range of [0, +∞)"

    total_sec = int(total_ms / 1000)
    millisecond = int(total_ms - total_sec * 1000)
    hour = int(total_sec / 3600)
    total_sec = total_sec - hour * 3600
    minute = int(total_sec / 60)
    second = int(total_sec - minute * 60)

    return hour, minute, second, millisecond

# for i in segments:
#     split_and_save(r"D:\birdsound\20211122\after de-noise.wav", r"D:\birdsound\20211122\after de-noise.wav", int(i[0]*1000), int(i[1]*1000))
#



