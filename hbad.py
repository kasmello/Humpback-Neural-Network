"""
Humpback Whale activity detection!
"""
import torchaudio
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

def calculate_energy_from_fft(fft_wav, low, high):
    total_amplitude = 0
    pass

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