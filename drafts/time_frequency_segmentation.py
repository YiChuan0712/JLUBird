# from scipy.io import wavfile
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA
#
# plt.figure(figsize=(21, 7))
# sample_rate, sig = wavfile.read(r"D:\birdsound\short(00-20-00-000)(00-20-20-000).wav")
#
# plt.subplot(211)
# plt.plot(sig)
#
# plt.subplot(212)
# spectrum, freqs, t, im = plt.specgram(sig, NFFT=512, noverlap=256, Fs=sample_rate, window=np.hamming(512))
# # plt.colorbar()
#
# pca = PCA(whiten=True)
# whitened = pca.fit_transform(spectrum)
#
# plt.show()

from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


sample_rate, sig = wavfile.read(r"D:\birdsound\kaggle1(00-00-00-000)(00-00-20-000).wav")

f, t, X = signal.stft(sig, fs=sample_rate, window='hann', nperseg=512, noverlap=256, nfft=None)
X = X.astype('float')

plt.subplot(211)
plt.pcolormesh(t, f, np.abs(X), cmap='Greys')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()



U, s, Vt = np.linalg.svd(X, full_matrices=False)
# U and Vt are the singular matrices, and s contains the singular values.
# Since the rows of both U and Vt are orthonormal vectors, then U * Vt
# will be white
X_white = np.dot(U, Vt)

plt.subplot(212)
plt.pcolormesh(t, f, np.abs(X_white), cmap='Greys')
plt.colorbar()
plt.title('STFT Magnitude - after whitening filter')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()


plt.show()

