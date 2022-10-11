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

def calculate_energy(arr, size):
    arr = arr.cpu().numpy()
    total = arr.sum()
    psd = total/(size**2)
    return psd

def process_and_predict(sound, dict_list, model, index_dict):
    i = 0
    update_sound = {'correct': 0, 'wrong': 0}
    update_blank = {'correct': 0, 'wrong': 0}
    wavform, sample_rate = torchaudio.load(sound)
    len_of_track = len(wavform[0])
    size_of_chunks = int(2.7 * sample_rate)  # in seconds
    sounds_for_this_file = deque()
    for t in range(0,len_of_track - size_of_chunks,int(0.1*sample_rate)):
        start = t
        end = (t + size_of_chunks)
        if i < len(dict_list) and dict_list[i]['Begin Path'] == sound:
            if  dict_list[i]['Beg File Samp (samples)'] <= end:
                sounds_for_this_file.appendleft(dict_list[i])
                i += 1
                print(f"STARTED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
            elif sounds_for_this_file[0]['End Time (s)'] <= start:
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
        Z = torch.tensor([[Z]],device=device, dtype=torch.float32)
        threshold = 0.4
        print(calculate_energy(Z, Z.size()[-1]))
        with torch.no_grad():
            output = F.softmax(model(Z),dim=1)
        percents = torch.topk(output.flatten(), 3).values.cpu().numpy()
        tensor = torch.topk(output.flatten(), 3).indices.cpu().numpy()
        label_tensor = [index_dict[code] for code in tensor]
        if len(sounds_for_this_file) == 0:
            if label_tensor[0] == 'Blank':
                update_blank['correct'] += 1
            else:
                update_blank['wrong'] += 1
        else:
            for item in sounds_for_this_file:
                if label_tensor[0] == item:
                    update_sound['correct'] += 1
                else:
                    update_sound['wrong'] += 1
        print(f'Top 3: {label_tensor}, PERCENTS: {percents}')

def run_through_audio(model_path, dict_path):
    model = load_model_for_training(model_path)
    index_dict = {}
    with open(dict_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_dict[int(row['Code'])] = row['Label']

    score_sound = {'correct': 0, 'wrong': 0}
    score_blank = {'correct': 0, 'wrong': 0}
    all_files = load_all_selection_tables()
    
    for file in all_files:
        all_sounds = load_all_sounds(file)
        dict_list = read_in_csv_times(file)
        
        for sound in all_sounds:
            try:
                update_sound, update_blank = process_and_predict(sound, dict_list, model, index_dict)
                for category in ['correct', 'wrong']:
                    score_sound[category] += update_sound[category]
                    score_blank[category] += update_blank[category]
            except KeyboardInterrupt:
                print('DONE')

