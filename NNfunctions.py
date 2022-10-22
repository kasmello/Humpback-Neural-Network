import torch
import torch.nn as nn
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
    boxes_per_page = 6
    for i in range(0,len(label_and_percent),boxes_per_page):
        fig, (ax1, ax2, ax3, ax4,ax5,ax6) = plt.subplots(1,boxes_per_page)
        fig.set_figheight(8)
        fig.set_figwidth(12)
        fig.suptitle('Boxplot of Energy for each Category')
        axs = [ax1,ax2,ax3,ax4,ax5,ax6]
        for j in range(boxes_per_page):
            try:
                axs[j].set_title(f'Category {label_and_percent[i+j][0]}\nMedian: {round(np.median(label_and_percent[i+j][1]),4)}',
                fontsize=7)
                axs[j].boxplot(label_and_percent[i+j][1],vert=True)
            except IndexError:
                break
        plt.show()
        plt.close()

def queue_noises(sounds_for_this_file,dict_list,end,i,sample_rate):
    done = False
    while not done:
        done=True
        if  dict_list[i]['Beg File Samp (samples)'] <= end:
            done=False
            sounds_for_this_file.appendleft(dict_list[i])
            print(f"QUEUED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
            i += 1
    return sounds_for_this_file,i

def dequeue_noises(sounds_for_this_file,start,sample_rate):
    i = 0
    length_of_queue = len(sounds_for_this_file)
    while i < length_of_queue:
        if sounds_for_this_file[i]['End Time (s)']*sample_rate <= start:
            popped = sounds_for_this_file.pop()
            print(f"DEQUEUED: {popped['Begin Path']} {popped['Selection']}")
            length_of_queue = len(sounds_for_this_file)
        else:
            i += 1
    return sounds_for_this_file

def process_and_predict(sound, dict_list, model, index_dict, start_time):
    i = 0
    update_sound = {'correct': 0, 'wrong': 0}
    update_blank = {'correct': 0, 'wrong': 0}
    update_table = []
    wavform, clean_wavform, sample_rate = grab_spectogram(sound)
    len_of_track = len(clean_wavform[0])
    dur = int(2.7 * sample_rate)  # in seconds
    sounds_in_this_current_window = deque()
    detection_values = {} #key = category, #values = dict
    for t in range(0,len_of_track - dur,int(0.2*sample_rate)):
        start = t
        curr_start_time = t + start_time
        end = (t + dur)
        if i < len(dict_list) and dict_list[i]['Begin Path'] == sound:
            sounds_in_this_current_window, i = queue_noises(sounds_in_this_current_window,dict_list,end,i,sample_rate)
            sounds_in_this_current_window = dequeue_noises(sounds_in_this_current_window,start,sample_rate)
        print(f'START SELECTION: {round(start/sample_rate,2)} END SELECTION: {round(end/sample_rate,2)}')
        Z = extract_wav(clean_wavform, sample_rate,start, dur)
        Z = resize(Z, (224,224),anti_aliasing=False)
        plt.imshow(Z,cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
        Z = torch.tensor([[Z]],device=device, dtype=torch.float32)
        energy_calculated = calculate_energy(clean_wavform[start:start+dur])
        if energy_calculated < 0.003:
            print('Detected as Blank due to low energy')
            if label_tensor[0] == 'Blank':
                update_blank['correct'] += 1
            else:
                update_blank['wrong'] += 1
            continue
        with torch.no_grad():
            output = F.softmax(model(Z),dim=1)
        percents = torch.topk(output.flatten(), 3).values.cpu().numpy()
        tensor = torch.topk(output.flatten(), 3).indices.cpu().numpy()
        label_tensor = [index_dict[code] for code in tensor]
        if len(sounds_in_this_current_window) == 0:
            if label_tensor[0] == 'Blank':
                update_blank['correct'] += 1
            else:
                update_blank['wrong'] += 1
        else:
            for item in sounds_in_this_current_window:
                if label_tensor[0] == item:
                    update_sound['correct'] += 1
                else:
                    update_sound['wrong'] += 1
        curr_start_seconds = curr_start_time/sample_rate
        if detection_values.get(label_tensor[0]):
            dict_to_update = detection_values[label_tensor[0]]
            dict_to_update['End Time (s)'] = curr_start_seconds+2.7
            dict_to_update['End File Samp (samples)'] = curr_start_time+2.7*sample_rate,
            detection_values[label_tensor[0]] = dict_to_update
        else:
            pop_list = [key for key in detection_values.keys() if key != label_tensor[0]]
            for code in pop_list:
                row = detection_values.pop(code)
                update_table.append(row)
            row = {
            'Begin Time (s)': curr_start_seconds,
            'End Time (s)': curr_start_seconds+2.7,
            'Beg File Samp (samples)': curr_start_time,
            'End File Samp (samples)': curr_start_time+2.7*sample_rate,
            'Code': label_tensor[0]
            }

            detection_values[label_tensor[0]] = row
        print(f'Top 3: {label_tensor}, PERCENTS: {round(percents,2)}')
    return update_sound, update_blank, update_table, start_time+len_of_track

def run_through_audio(model_path, dict_path):
    index_dict = {}
    table_dict = []
    with open(dict_path,'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_dict[int(row['Code'])] = row['Label']
    model = load_model_for_training(model_path,len(index_dict)) #assigning the correct model...
    score_sound = {'correct': 0, 'wrong': 0}
    score_blank = {'correct': 0, 'wrong': 0}
    all_tables = load_all_selection_tables()
    start_time = 0
    try:
        for table in all_tables:
            all_sounds = load_all_sounds(table)
            dict_list = read_in_csv_times(table)
            for sound in all_sounds[:50]:
                update_sound, update_blank, update_table, start_time = process_and_predict(sound, dict_list, model, index_dict,start_time)
                table_dict.extend(update_table)
                for category in ['correct', 'wrong']:
                    score_sound[category] += update_sound[category]
                    score_blank[category] += update_blank[category]
        save_string = 'saved_prediction.txt'
        print(f'SAVING AS {save_string}')
        save_file(table_dict,'saved_prediction.txt')
    except KeyboardInterrupt:
        print('DONE')

def find_file(path,search_string):
    """
    find file with specific extension
    input:
        extension - string of extension (e.g. csv)
    output:
        str of file path
    """
    all_files = sorted(pathlib.Path(path).glob(search_string))
    ask_list = []
    for index, file_path_obj in enumerate(all_files):
        file_path = file_path_obj.as_posix()
        name = file_path.split('/')[-1]
        ask_list.append([f'{index}: {name}',file_path])
    ask_str = ''
    for item in ask_list:
        ask_str += item[0]+'\n'
    index_for_model = input(f'Type number to select a file: 0 to {len(ask_list)-1}\n{ask_str}')
    return ask_list[int(index_for_model)][1]

def save_file(table_dict, save_str):
    """
    Saves selection table for raven to open
    """
    if len(table_dict) <= 0:
        print('Nothing detected!')
        return None
    with open(save_str, 'w') as f:
        fieldnames = [table_dict[0].keys()]
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_dict)

def signaltonoise_dB(a):
    """
    a is the wav array
    APPARENTLY THIS IS NOT CORRECT
    """
    try:
        m = torch.mean(a).item()
        sd = torch.std(a).item()
    except TypeError:
        m = np.mean(a)
        sd = np.std(a)
    return 10*np.log10(abs(np.where(sd == 0, 0, m/sd)))

def get_model_from_name(model_name,num_labels):
    if model_name[:3].lower()=='net':
        return Net(num_labels).to(device)

    elif model_name[:5].lower()=='cnnet':
        return CNNet(num_labels).to(device)

    elif model_name[:8].lower()=='resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_labels)
        model = model.to(device)
        return model

    elif model_name[:5].lower()=='vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        #print(model) to look at structure
        model.classifier.add_module('7', nn.ReLU())
        model.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
        model.classifier.add_module('9', nn.Linear(1000, num_labels))
        model = model.to(device)
        return model

    elif model_name[:3].lower()=='vit':
        return timm.create_model('vit_small_patch16_224',pretrained=True, num_classes=num_labels, in_chans=1).to(device)