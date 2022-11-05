"""
Humpback Whale activity detection!
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import butter, lfilter
from read_audio_module import grab_wavform

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a    

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_db(f,freq, low, high):
    db = 10 * np.log10(np.sum(abs(f[(freq>=low) & (freq<=high)])**2))
    return np.where(db<0,0,db)

def calculate_energy_from_fft(wav, mid_point, sample_rate):
    try:
        wav = wav[0].numpy()
    except AttributeError:
        pass
    low = 50
    high = 3000
    shorttime = int(sample_rate/4) #quarter of a second
    temp_wav = wav[:shorttime]
    f = np.fft.fft(temp_wav)
    freq = np.fft.fftfreq(len(temp_wav), d=1/sample_rate)
    freq = freq[:len(freq)//2]
    f = f[:len(f)//2]

    d1 = get_db(f,freq,low, mid_point)
    d2 = get_db(f,freq,mid_point, high)
    return d1,d2

def calculate_noise_ratio(d1,d2):
    try:
        ratio = np.where(d1>d2,d2/d1,d1/d2)
        return ratio
    except FloatingPointError:
        return 0

def calculate_energy(wav,lowcut=1, highcut=2990,sample_rate=10000):
    """
    Calculates energy based on frequency segment
    """
    try:
        wav = wav[0].numpy()
    except AttributeError:
        wav = wav
    y = fft(wav)
    amps = butter_bandpass_filter(wav, lowcut, highcut, sample_rate, 3)
    energy = sum([x*2 for x in amps])
    return abs(energy)