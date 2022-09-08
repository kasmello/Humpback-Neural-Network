import torch
import torchaudio
import pathlib
import platform
import scipy
import math
import os
import numpy as np
import torchaudio.functional as F
import torchaudio.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from transformclasses import normalise

selection_table_locations = [
    '/HumpbackDetect/Humpback Units/Selection Tables/',
    '/HumpbackDetect/Minke Boings/3187 IMOS WA NW SNR4/Selection Tables/'
]

file_count_tracker = {}

switch_dictionary = {
    'LMw': 'LM',
    'LMh': 'LM',
    'LMvu': 'LMu',
    'MStatic': 'Blank',
    'SStatic': 'Blank',
    'N3':'N4',
    'W': 'N6',
    'Dw': 'N6',
    'IG': 'lG',
    'llG': 'lG',
    'Dm': 'D',
    'L': 'Dh',
    'G': 'D',
    'Hi': 'J',
    'FSU': 'Fsu',
    'FSu': 'Fsu',
    'DvH': 'Dvh',
    '1': 'MB',
}

def find_root():
    """
    simple function to return string of data folder
    """
    if platform.system() == 'Darwin':  # MAC
        return '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
    elif platform.system() == 'Windows':
        return 'C://Users/Karmel 0481098535/Desktop/Humpback'

ROOT = find_root()

def change_to_pc_or_mac():
    if platform.system() == 'Darwin':  # MAC
        return [f'/Volumes/HD{dir}' for dir in selection_table_locations]
    elif platform.system() == 'Windows':
       return [f'D:{dir}' for dir in selection_table_locations]

def save_image(Z, label):
    label = str(label)
    label = switch_dictionary.get(label,label)
    folder_path = os.path.join(ROOT,label)
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        pass
    file_count_tracker[label] = file_count_tracker.get(label,0) + 1
    save_name = f'{label}-{file_count_tracker[label]}.png'
    plt.imsave(os.path.join(folder_path,save_name), Z, cmap='gray')
    print(f'SAVED {save_name}')    

def pad_start_and_end(start,dur, sample_rate):
    segment_dur = 2.7 * sample_rate
    if dur < segment_dur:
        padding = (segment_dur - dur) / 2
        if start < padding:
            start = 0
        else:
            start = start - padding
    else:
        tocut  = (dur - segment_dur)/ 2
        start = start + tocut
    end = start + segment_dur
    return math.floor(start), math.floor(end)

def extract_wav(wav_dir, start, dur):
    waveform, sample_rate = torchaudio.load(wav_dir)
    start, end = pad_start_and_end(start,dur, sample_rate)
    waveform = waveform[:,start:end+1]
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    NFFT=1024
    Pxx, freqs, bins, im = plt.specgram(waveform[0], Fs=sample_rate, NFFT=NFFT, noverlap=NFFT/2,
    window=np.hanning(NFFT))
    Pxx = Pxx[(freqs >= 50) & (freqs <= 3000)]
    freqs = freqs[(freqs >= 50) & (freqs <= 3000)]
    Z = normalise(Pxx)
    return Z

def read_wav_segment(wav_dir, start, dur, label):
    Z = extract_wav(wav_dir, start, dur)
    Z = resize(Z, (224,224),anti_aliasing=False)
    plt.imshow(Z, cmap='gray')
    save_image(Z, label)

    

def open_up_wavs(file):
    try:
        dt = pd.read_table(file)
    except UnicodeDecodeError:
        dt = pd.read_table(file, encoding='iso-8859-1')
    if dt.shape[0] > 0 and file.as_posix().split('/')[-1][0:2] != '._':

        print(file.as_posix())
        print(file)
        [read_wav_segment(wav_dir, start, dur, label) for wav_dir, start, dur, label in zip(dt['Begin Path'], dt['Beg File Samp (samples)'],
        dt['Sample Length (samples)'], dt['Code'])]

def wav_to_spec():

    selection_table_locations = change_to_pc_or_mac()
    for location in selection_table_locations:
        files = pathlib.Path(location).glob('*.txt')
        for file in files:
            open_up_wavs(file)

if __name__ == '__main__':

    wav_to_spec()

    