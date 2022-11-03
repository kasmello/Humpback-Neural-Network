import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
from skimage.transform import resize
from transformclasses import normalise

beta = 1
samples = int(6000*2.7)
NFFT=1024
sample_rate=6000
A = cn.powerlaw_psd_gaussian(beta, samples)
Pxx, freqs, bins, im = plt.specgram(A, Fs=sample_rate, NFFT=NFFT, noverlap=NFFT/2,
    window=np.hanning(NFFT))
Pxx = Pxx[(freqs >= 50) & (freqs <= 3000)]
freqs = freqs[(freqs >= 50) & (freqs <= 3000)]
Z = normalise(Pxx, convert=False,fix_range=False)
Z = resize(Z, (224,224),anti_aliasing=False)
plt.imshow(Z,cmap='gray')
plt.axis('off')
plt.show()
plt.close()

#Ploting first subfiure
# plt.plot(A, color='black', linewidth=1)
# plt.title('Colored Noise for β='+str(beta))
# plt.xlabel('Samples (time-steps)')
# plt.ylabel('Amplitude(t)', fontsize='large')
# plt.show()