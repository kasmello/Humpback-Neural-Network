import torch
import pathlib
import numpy as np
import csv
import math
import timm
import platform
import torchaudio
from read_audio_module import extract_wav
from scipy.io import wavfile
from collections import deque
from skimage.transform import resize
from pattern_recognition import pattern_analyis
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms
from pydub import AudioSegment 
from NNclasses import Net, CNNet

device = torch.device("cpu" if platform.system()=='Windows'
                                else "mps")

def pad_out_row(row, segment_dur):
    dur = float(row['End Time (s)'])-float(row['Begin Time (s)'])
    if dur <= segment_dur:
        padding = (segment_dur - dur) / 2
        if float(row['File Offset (s)']) < padding:
            row['Begin Time (s)'] = 0
        else:
            row['Begin Time (s)'] = float(row['Begin Time (s)']) - math.floor(padding)
    else:
        tocut  = math.ceil((dur - segment_dur)/ 2)
        row['Begin Time (s)'] = float(row['Begin Time (s)']) + tocut
    row['End Time (s)'] = float(row['Begin Time (s)']) + segment_dur
    return row
        

def read_in_csv_times(file):
    with open(file, 'r') as csv_file:
        csv_dict = csv.DictReader(csv_file,delimiter=',')
        for d in csv_dict:
            d['Selection'] = d.pop('\ufeffSelection')
        dict_list = []
        for i, row in enumerate(csv_dict):
            dict_list.append(pad_out_row(row, 2.7))
        return dict_list


def wav_to_spectogram(item, save = True):
    fs, x = wavfile.read(item)
    Pxx, freqs, bins, im = plt.specgram(x, Fs=fs,NFFT=1024)
    Pxx = Pxx[(freqs >= 50) & (freqs <= 3000)]
    log_Pxx = 10*np.log10(np.flipud(Pxx))
    log_Pxx = resize(log_Pxx, (224,224),anti_aliasing=False)
    max_box = np.max(log_Pxx)
    if max_box < 0:
        max_box=0
    
    for i, row in enumerate(log_Pxx):
        for j, item in enumerate(row):
            if log_Pxx[i][j] < 0:
                log_Pxx[i][j] = 0
            if log_Pxx[i][j] > max_box-5:
                log_Pxx[i][j] = max_box - 5
            elif log_Pxx[i][j] < max_box-20:
                # log_Pxx[i][j] = max_box-20
                # print(log_Pxx[i][j], max_box-20) 
                pass

    min_value = np.min(log_Pxx)
    max_value = np.max(log_Pxx)
    for i, row in enumerate(log_Pxx):
        for j, item in enumerate(row):
            log_Pxx[i][j] = (item - min_value)/max_value

    # plt.pcolormesh(bins, freqs, 10*np.log10(Pxx))
    plt.imshow(log_Pxx,cmap='gray')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()
    if save:
        plt.savefig(f'{item[:-4]}.png',bbox_inches=0)
        print('saved!')

    return torch.tensor([log_Pxx], device=device, dtype=torch.float32)

def load_all_selection_tables():
    """
    simple function to return location of all selection tables
    """
    file_dir =  pathlib.Path('/Volumes/Karmel TestSets/Selection Tables/').glob('*.csv')
    return [item.as_posix() for item in file_dir]

def load_all_sounds(file):
    """
    simple function to return localtion of all sounds
    input:
        file - str of location of selection table - this is used to find the sounds
    """
    file_name = str(file).split('/')[-1]
    sounds_dir = pathlib.Path('/Volumes/Karmel TestSets/').glob(f'{file_name[0:5]}*/wavfiles/*.wav')
    return [sound.as_posix() for sound in sounds_dir]

def get_model_from_name(model_name):
    if model_name[:3].lower()=='net':
        return Net().to(device)
    elif model_name[:5].lower()=='cnnet':
        return CNNet().to(device)
    elif model_name[:8].lower()=='resnet18':
        return models.resnet18(pretrained=True).to(device)
    elif model_name[:5].lower()=='vgg16':
        return models.vgg16(pretrained=True).to(device)
    elif model_name[:3].lower()=='vit':
        return timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=25, in_chans=1).to(device)

def load_model_for_training(model_path):
    model_name = model_path.split('/')[-1][:-3]
    model = get_model_from_name(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device)) #investigate strict
    model.eval()
    return model

def run_through_audio(model_path, dict_path):
    model = load_model_for_training(model_path)
    index_dict = {}
    # pa = pattern_analyis()
    with open(dict_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_dict[int(row['Code'])] = row['Label']

    all_files = load_all_selection_tables()
    for file in all_files:
        all_sounds = load_all_sounds(file)
        dict_list = read_in_csv_times(file)
        i = 0
        for sound in all_sounds:
            wavform, sample_rate = torchaudio.load(sound)
            len_of_track = len(wavform[0])
            size_of_chunks = int(2.7 * sample_rate)  # in seconds
            sounds_for_this_file = deque()
            for t in range(0,len_of_track - size_of_chunks,int(0.1*sample_rate)):
                start = t
                end = (t + size_of_chunks)
                if i < len(dict_list) and dict_list[i]['Begin Path'] == sound:
                    if  dict_list[i]['Beg File Samp (samples)'] >= start:
                        sounds_for_this_file.appendleft(dict_list[i])
                        i += 1
                        print(f"STARTED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
                    elif sounds_for_this_file[0]['End Time (s)'] <= end:
                        popped = sounds_for_this_file.pop()
                        print(f"ENDED: {popped[i]['Begin Path']} {popped[i]['Selection']}")
                        
                print(f'START: {round(start/sample_rate,2)} END: {round(end/sample_rate,2)}')
                Z = extract_wav(sound, start, size_of_chunks)
                Z = resize(Z, (224,224),anti_aliasing=False)
                plt.imshow(Z,cmap='gray')
                plt.axis('off')
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
                Z = torch.tensor([[Z]],device=device)
                with torch.no_grad():
                    output = F.log_softmax(model(Z),dim=1)
                prediction = index_dict[torch.argmax(output).item()]
                print(f'PREDICTION: {prediction}')
                # pa.add_prediction(prediction, start, end, file, sound)

def run_through_audio_scrap(model_path, dict_path):
    model = load_model_for_training(model_path)
    index_dict = {}
    pa = pattern_analyis()
    with open(dict_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_dict[int(row['Code'])] = row['Label']

    all_files = load_all_selection_tables()
    for file in all_files:
        all_sounds = load_all_sounds(file)
        dict_list = read_in_csv_times(file)
        i = 0
        for sound in all_sounds:
            wav_file = AudioSegment.from_wav(sound)
            len_of_track = len(wav_file)
            size_of_chunks = int(2.7 * 1000)  # in seconds
            sounds_for_this_file = deque()
            for t in range(0,len_of_track - size_of_chunks,100):
                start = t
                end = (t + size_of_chunks)
                if i < len(dict_list) and dict_list[i]['Begin Path'] == sound:
                    if  dict_list[i]['Begin Time (s)'] >= start/1000:
                        sounds_for_this_file.appendleft(dict_list[i])
                        i += 1
                        print(f"STARTED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
                    elif sounds_for_this_file[0]['End Time (s)'] <= end/1000:
                        popped = sounds_for_this_file.pop()
                        print(f"ENDED: {popped[i]['Begin Path']} {popped[i]['Selection']}")
                        
                print(f'START: {start} END: {end}')
                segment = wav_file[start:end]
                temp_file = pathlib.Path('/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav')
                segment.export(
                    temp_file.as_posix(), format='wav')
                array_to_predict = wav_to_spectogram(
                    temp_file.as_posix(), save=False)
                with torch.no_grad():
                    output = F.log_softmax(model(array_to_predict.float()),dim=1)
                prediction = index_dict[torch.argmax(output).item()]
                print(f'PREDICTION: {prediction}')
                pa.add_prediction(prediction, start, end, file, sound)
                temp_file.unlink()
        





