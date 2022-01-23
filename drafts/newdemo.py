import demo_head as dh
import matplotlib.pyplot as plt
import librosa
import wave
import numpy as np
import scipy
import os

filename = r"D:\birdsound\20211215\short(00-20-00-000)(00-20-05-000).wav"
#filename = r"D:\birdsound\20211215\kaggle1.wav"

samplerate, wave = dh.read_wave_file_not_normalized(filename)

X = librosa.stft(wave.astype('float'), n_fft=512, hop_length=128, win_length=512)

print(X.shape)

X = np.abs(X) ** 2

print(X.shape)

def openAudioFile(path, sample_rate=48000, offset=0.0, duration=None):
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')
    return sig, rate

sig, rate = openAudioFile(filename)
f, t, spec = scipy.signal.spectrogram(sig,
                                          fs=rate,
                                          noverlap=128,
                                          nfft=512,
                                          )
print(rate)

print(len(f))
print(len(t))
print(f)
# print(t)

print(spec.shape)

print(X[0][0])
print(spec[0][0])

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

fig = plt.figure(1)
subplot_image(X, 411, "Spectrogram")


fig = plt.figure(1)
subplot_image(spec, 412, "Spectrogram")

plt.subplot(413)
plt.pcolormesh(t, f, np.abs(spec), cmap='Greys')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
#plt.show()
