import demo_head as dh
import matplotlib.pyplot as plt
import librosa
import wave
import numpy as np
filename = r"D:\birdsound\20211215\short(00-20-00-000)(00-20-05-000).wav"

# samplerate, wave = dh.read_wave_file_not_normalized(filename)
# print(wave.shape)
#
# print(samplerate)
# print(wave.shape)
#
# sp = dh.wave_to_complex_spectrogram(wave, samplerate)
#
# sp2 = dh.wave_to_amplitude_spectrogram(wave, samplerate)
#
# print(sp.shape)
# print(sp2.shape)

dh.sprengel_binary_mask_from_wave_file(filename)

#dh.preprocess_sound_file(filename, r"D:\birdsound\20211215\k1sound", r"D:\birdsound\20211215\k1noise", 3)
#
# fs, x = dh.read_wave_file(filename)
# Sxx = dh.wave_to_amplitude_spectrogram(x, fs)
# Sxx_log = dh.wave_to_log_amplitude_spectrogram(x, fs)
# #
# def grayify_cmap(cmap):
#     """Return a grayscale version of the colormap"""
#     cmap = plt.cm.get_cmap(cmap)
#     colors = cmap(np.arange(cmap.N))
#
#     # convert RGBA to perceived greyscale luminance
#     # cf. http://alienryderflex.com/hsp.html
#     RGB_weight = [0.299, 0.587, 0.114]
#     luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
#     colors[:, :3] = luminance[:, np.newaxis]

#    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)
#
# def subplot_image(Sxx, n_subplot, title):
#     cmap = grayify_cmap('cubehelix_r')
#     cmap = plt.cm.get_cmap('jet')
#     cmap = grayify_cmap('jet')
#     plt.subplot(n_subplot)
#     plt.title(title)
#     plt.pcolormesh(Sxx, cmap=cmap)
#
# # plot spectrogram
# fig = plt.figure(1)
# subplot_image(Sxx_log, 411, "Spectrogram")
plt.show()