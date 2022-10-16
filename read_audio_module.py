import torch
import torchaudio
import pathlib
import platform
import math
import os
import csv
import numpy as np
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import noisereduce as nr
from tqdm import tqdm
from random import sample
from skimage.transform import resize
from transformclasses import normalise

selection_table_locations = [
    '/HumpbackDetect/Humpback Units/Selection Tables/',
    '/HumpbackDetect/Minke Boings/Chosen Selection Tables/'
]


progress_table = {}

switch_dictionary = {
    'LMw': 'LM',
    'LMh': 'LM',
    'LMvu': 'LMu',
    'MStatic': 'Blank',
    'SStatic': 'Blank',
    'N3':'N4',
    'W': 'N6',
    'Dw': 'N4',
    'IG': 'lG',
    'llG': 'lG',
    'Dm': 'G',
    'L': 'Dh',
    'Hi': 'LJ',
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

def save_image(Z, state,label):
    folder_path = os.path.join(ROOT,state,label)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
    save_name = f'{label}-{progress_table.get(label,0)+1}.png'
    plt.imsave(os.path.join(folder_path,save_name), Z, cmap='gray')
    # if state=='dirty':
    #     print(f'SAVED {save_name}')    

def pad_start_and_end(start,dur, sample_rate):
    segment_dur = 2.7 * sample_rate
    start = float(start)
    dur = float(dur)
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

def normalise_waveform(wav):
    mean = wav.mean()
    std = wav.std()
    wav = (wav-mean)/std
    return wav
    


def extract_wav(wav, sample_rate, start, dur):
    start, end = pad_start_and_end(start,dur, sample_rate)
    wav = wav[:,start:end+1]
    # waveform = waveform.numpy()
    # (S, f) = plt.psd(waveform[0], Fs=sample_rate)
    # plt.semilogy(f, S)
    # plt.xlim([0, 100])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    # plt.show()
    # num_channels, num_frames = wav.shape
    # time_axis = torch.arange(0, num_frames) / sample_rate
    NFFT=1024
    Pxx, freqs, bins, im = plt.specgram(wav[0], Fs=sample_rate, NFFT=NFFT, noverlap=NFFT/2,
    window=np.hanning(NFFT))
    Pxx = Pxx[(freqs >= 50) & (freqs <= 3000)]
    freqs = freqs[(freqs >= 50) & (freqs <= 3000)]
    Z = normalise(Pxx)
    return Z

def grab_spectogram(wav_dir):
    wavform, sample_rate = torchaudio.load(wav_dir)
    clean_wavform = nr.reduce_noise(y=wavform, sr=sample_rate, time_mask_smooth_ms=64, stationary=False)
    return wavform, clean_wavform, sample_rate

def load_in_progress():
    with open('progress.csv','r') as progress_file:
        all_lines = progress_file.read().splitlines()
        for line in all_lines:
            line_list = line.split(',')
            try:
                insert = int(line_list[1])
            except ValueError:
                insert = line_list[1]
            progress_table[line_list[0]] = insert
    print(progress_table)

def get_selection_table_name(filepath):
    return filepath.split('/')[-1]

def write_progress_table():
    with open('progress.csv','w') as progress_file:
        for row, value in progress_table.items():
            progress_file.write(f"{row},{value}\n")

def run_through_file(filename, dt, progress_table,region_name=None):
    """
    runs through 1 whole selection table
    """
    wavstring = ''
    clean_wavform = None
    wavform = None
    sample_rate = 0
    if region_name: progress_table[filename] = progress_table.get(filename,0)
    for row in tqdm(range(len(dt))):
        wav_dir = dt[row]['Begin Path']
        start = dt[row]['Beg File Samp (samples)']
        dur = dt[row]['Sample Length (samples)']
        label = str(dt[row]['Code'])
        label = switch_dictionary.get(label,label)
        if region_name:
            if row < progress_table[filename]:
                continue #iterating to the correct row
        if wav_dir != wavstring:
            wavstring = wav_dir
            wavform, clean_wavform, sample_rate = grab_spectogram(wav_dir)
        for index, wav in enumerate([wavform, clean_wavform]):
            Z = extract_wav(wav, sample_rate, start, dur)
            Z = resize(Z, (224,224),anti_aliasing=False)
            plt.imshow(Z, cmap='gray')
            if index == 0:
                save_image(Z, 'dirty',label)
                if region_name:
                    progress_table[label] = progress_table.get(label,0) + 1
                    progress_table[region_name] = progress_table.get(region_name,0) + 1
                    progress_table['Selection Row'] = progress_table.get('Selection Row', 0) + 1
                    progress_table[filename] = progress_table.get(filename,0) + 1
                    write_progress_table()
            else:
                save_image(Z, 'clean',label)
    return progress_table
                    

def wav_to_spec():
    load_file = False
    if os.path.exists('progress.csv'):
        load_in_progress()
        load_file=True
    selection_table_locations = change_to_pc_or_mac()
    for location in selection_table_locations:
        files = pathlib.Path(location).glob('*.txt')
        for file in files:
            filename = get_selection_table_name(file.as_posix())
            if filename in ['t3152 Sep.txt'] or filename[0:2] == '._':
                continue
            region_name = filename.split()[0]
            if load_file:
                if filename == progress_table['Selection Table']:
                    load_file = False
                else:
                    print('filename')
                    print(filename)
                    print('progress_table')
                    print(progress_table['Selection Table'])
                    continue
            progress_table['Selection Table'] = filename

            try:
                with open(file,'r') as selection_table:
                    dt = list(csv.DictReader(selection_table,delimiter='\t'))
            except UnicodeDecodeError:
                with open(file,'r',encoding='iso-8859-1') as selection_table:
                    dt = list(csv.DictReader(selection_table,delimiter='\t'))
            
            progress_table = run_through_file(filename, dt, progress_table,region_name)
                    
    
   


if __name__ == '__main__':

    wav_to_spec()

    