import torch
import pathlib
import numpy as np
import csv
import math
import platform
from read_audio_module import extract_wav, grab_spectogram
from scipy.io import wavfile
from tqdm import tqdm
from collections import deque
from skimage.transform import resize
from pattern_recognition import pattern_analyis
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from NNclasses import Net, CNNet
from NNtrainingfunctions import get_model_from_name

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
        csv_dict = csv.DictReader(csv_file,delimiter='\t')
        # for d in csv_dict:
        #     d['Selection'] = d.pop('\ufeffSelection')
        dict_list = []
        for i, row in enumerate(csv_dict):
            dict_list.append(pad_out_row(row, 2.7))
        return dict_list

def load_all_selection_tables():
    """
    simple function to return location of all selection tables
    """
    file_dir =  pathlib.Path('/Volumes/HD/HumpbackDetect/Humpback Units/Selection Tables/t3152 Sep.txt')
    return [file_dir]

def load_all_sounds(file):
    """
    simple function to return localtion of all sounds
    input:
        file - str of location of selection table - this is used to find the sounds
    """
    file_name = str(file).split('/')[-1]
    sounds_dir = pathlib.Path('/Volumes/HD/HumpbackDetect/Humpback Units/').glob(f'{file_name[0:5]}*/WAV/201209*.wav')
    return [sound.as_posix() for sound in sounds_dir]

def load_model_for_training(model_path, num_labels):
    model_name = model_path.split('/')[-1][:-3]
    model = get_model_from_name(model_name,num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device)) #investigate strict
    model.eval()
    return model

def calculate_energy(arr):
    arr = arr.cpu().numpy()
    total = arr.sum()
    psd = total/(arr.size)
    return psd

def get_psd_distribution_of_all_images(DATA):
    psd_dict = {}
    for batch in tqdm(DATA.all_training):
        x, y = batch
        for img, lbl in zip(x,y):
            psd_dict[DATA.inverse_encode([lbl])] = psd_dict.get(DATA.inverse_encode([lbl]),[]) + [calculate_energy(img)]
    label_and_percent = list(psd_dict.items())
    # for label, percent in label_and_percent:
    #     print(f"{label}: {round(np.mean(percent),5)}")
    for i in range(0,len(label_and_percent),5):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
        fig.suptitle('Boxplot of energy')
        axs = [ax1,ax2,ax3,ax4,ax5]
        for j in range(5):
            try:
                axs[j].set_title(f'Category {label_and_percent[i+j][0]}\nMedian: {round(np.median(label_and_percent[i+j][1]),5)}')
                axs[j].boxplot(label_and_percent[i+j][1],vert=False)
            except IndexError:
                break
        plt.show()



def process_and_predict(sound, dict_list, model, index_dict):
    i = 0
    update_sound = {'correct': 0, 'wrong': 0}
    update_blank = {'correct': 0, 'wrong': 0}
    wavform, clean_wavform, sample_rate = grab_spectogram(sound)
    len_of_track = len(clean_wavform[0])
    dur = int(2.7 * sample_rate)  # in seconds
    sounds_for_this_file = deque()
    for t in range(0,len_of_track - dur,int(0.1*sample_rate)):
        start = t
        end = (t + dur)
        if i < len(dict_list) and dict_list[i]['Begin Path'] == sound:
            if  dict_list[i]['Beg File Samp (samples)'] <= end:
                sounds_for_this_file.appendleft(dict_list[i])
                i += 1
                print(f"STARTED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
            elif sounds_for_this_file[0]['End Time (s)'] <= start:
                popped = sounds_for_this_file.pop()
                print(f"ENDED: {popped[i]['Begin Path']} {popped[i]['Selection']}")
        print(f'START: {round(start/sample_rate,2)} END: {round(end/sample_rate,2)}')
        Z = extract_wav(clean_wavform, sample_rate,start, dur)
        Z = resize(Z, (224,224),anti_aliasing=False)
        plt.imshow(Z,cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        Z = torch.tensor([[Z]],device=device, dtype=torch.float32)
        threshold = 0.4
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
        print(f'Top 3: {label_tensor}, PERCENTS: {round(percents,2)}')

def run_through_audio(model_path, dict_path):
    index_dict = {}
    with open(dict_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_dict[int(row['Code'])] = row['Label']
    model = load_model_for_training(model_path,len(index_dict))

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
