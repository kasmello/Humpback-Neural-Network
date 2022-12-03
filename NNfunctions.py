import torch
import torch.nn as nn
import pathlib
import numpy as np
import csv
import timm
import os
import math
import platform
import wandb
import gc
from datetime import date
from read_audio_module import extract_wav, grab_wavform
from scipy.io import wavfile
from tqdm import tqdm
from collections import deque
from skimage.transform import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from unsupervised_vad import grab_sound_detection_areas
from NNclasses import Net, CNNet
from transformclasses import normalise
from hbad import calculate_energy, calculate_energy_from_fft, calculate_noise_ratio

device = torch.device("cuda" if platform.system()=='Windows'
                                else "mps")

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

selection_stream_pair = {'t3152 Sep.txt':'/Volumes/HD/HumpbackDetect/Humpback Units/t3152 981 RPS BHP Port Hedland SNR47/WAV/20120901000001.wav',
'2845 August 13-16.txt': '/Volumes/HD/HumpbackDetect/Humpback Units/t2845Backup/WAV/20090813000001.wav'}

blank_sound_pair = {'nosound.txt': '/Volumes/HD/HumpbackDetect/Humpback Units/NOSOUND/20100502040001.wav',
'nosound copy.txt': '/Volumes/HD/HumpbackDetect/Humpback Units/NOSOUND/20100502041500.wav'}

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
            if row['Begin Path'] != selection_stream_pair[file.name]:
                break
            dict_list.append(pad_out_row(row, 2.7))
        return dict_list

def load_all_selection_tables():
    """
    simple function to return location of all selection tables
    """
    all_dir = []
    selection_table_dir = '/Volumes/HD/HumpbackDetect/Humpback Units/Selection Tables'
    for selection_table in selection_stream_pair.keys():
        all_dir.append(pathlib.Path(os.path.join(selection_table_dir,selection_table)))
    return all_dir

def load_all_sounds(file):
    """
    simple function to return localtion of all sounds
    input:
        file - str of location of selection table - this is used to find the sounds
    """
    file_name = file.name
    sounds_dir =selection_stream_pair[file_name]
    return sounds_dir

def load_model_for_training(model_path, num_labels):
    model_name = model_path.split('/')[-1][:-4]
    model = get_model_from_name(model_name,num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device)) #investigate strict
    model.eval()
    return model


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

def queue_noises(sounds_for_this_file,dict_list,end,i):
    done = False
    count_queued = 0
    while not done:
        done=True
        if i > len(dict_list):
            break
        if  int(dict_list[i]['Beg File Samp (samples)']) <= end:
            if dict_list[i]['Code'] not in ['Blank','',\
                '1a','','Fu','YSh','YS','FC','FP','V','Sqh','ll']:
                done=False
                dict_list[i]['Code'] = switch_dictionary.get(dict_list[i]['Code'],dict_list[i]['Code'])
                dict_list[i]['found']='notfound'
                sounds_for_this_file.appendleft(dict_list[i])
                count_queued += 1
            # print(f"QUEUED: {dict_list[i]['Begin Path']} {dict_list[i]['Selection']}")
            i += 1
            

    return sounds_for_this_file,i,count_queued

def dequeue_noises(sounds_for_this_file,start,sample_rate):
    i = 0
    update_list = [] #(misclassified/classified, category)
    length_of_queue = len(sounds_for_this_file)
    while i < len(sounds_for_this_file):
        if float(sounds_for_this_file[i]['End Time (s)'])*sample_rate <= start:
            popped = sounds_for_this_file.pop()
            list_to_append = [popped['found'],popped['Code']]
            if popped.get('reason'):
                list_to_append.append(popped['reason'])
            # print(f"DEQUEUED: {popped['Begin Path']} {popped['Selection']}")
            update_list.append(list_to_append)
        else:
            i += 1
    
    return sounds_for_this_file,update_list

def check_if_blank_on_top_and_over_threshold(tensors,percents,threshold):
    if tensors[0]=='Blank' and percents[0]>threshold:
        return True
    return False


def process_and_predict(sound, scores, dict_list, model, index_dict, vad_threshold, percentage_threshold,topk):
    i = 0
    update_table = []
    wav, clean_wavform, sample_rate = grab_wavform(sound)
    vad_arr = grab_sound_detection_areas(wav, sample_rate, vad_threshold)
    #heckin plot some spectrograms here pls
    len_of_track = len(wav[0])
    dur = int(2.7 * sample_rate)  # in seconds
    detection_dur = int(0.2*sample_rate)
    sounds_in_this_current_window = deque()
    detection_values = {} #key = category, #values = dict
    for t in tqdm(range(0,len_of_track - dur,detection_dur)):
        reason=None
        blank = False
        start = t
        curr_start_time = t
        end = t + detection_dur
        if end >= len_of_track-dur:
            break
        if sum(vad_arr[start:end]) < sample_rate*0.05:
            blank = True
            reason='blank_reason_vad'
            tensor_percents = []
        try:
            if i < len(dict_list):
                sounds_in_this_current_window, i,count_queued = queue_noises(sounds_in_this_current_window,dict_list,end,i)
                scores['table_sounds'] += count_queued
                sounds_in_this_current_window, update_list = dequeue_noises(sounds_in_this_current_window,start,sample_rate)
                for item in update_list:
                    if item[0]=='found':
                        scores['sounds_classified'][item[1]] = scores['sounds_classified'].get(item[1],0)+1
                    elif item[0]=='blank':
                        scores['sounds_misclassified_blank'][item[1]] = scores['sounds_misclassified_blank'].get(item[1],0)+1
                        scores[item[2]] += 1
                    elif item[0]=='notfound':
                        scores['sounds_misclassified'][item[1]] = scores['sounds_misclassified'].get(item[1],0)+1


        except IndexError:
            pass


        # print(f'START SELECTION: {round(start/sample_rate,2)} END SELECTION: {round(end/sample_rate,2)}')
        Z = extract_wav(wav, sample_rate,start, dur)
        Z = normalise(Z,convert=True,fix_range=False)
        Z = resize(Z, (224,224),anti_aliasing=False)
        # plt.imshow(Z,cmap='gray')
        # plt.axis('on')
        # plt.show(block=False)
        # plt.pause(0.1)
        # plt.close()
        if model:
            if not blank:
                Z = torch.tensor([[Z]],device=device, dtype=torch.float32)
                with torch.no_grad():
                    output = F.softmax(model(Z),dim=1)
                percents = torch.topk(output.flatten(), topk).values.cpu().numpy()
                tensors = torch.topk(output.flatten(), topk).indices.cpu().numpy()
                label_tensor = [index_dict[code] for code in tensors]
                
                if tensors[0]!='Blank':
                    tensor_percents = [(tensor,percent) for tensor, percent in zip(label_tensor,percents) if percent >= percentage_threshold and tensor != 'Blank']
                    
                    if len(tensor_percents) == 0:
                        blank = True
                        blank='blank_reason_below_thresh'
                else:
                    blank=True
                    tensor_percents = []
                    reason='blank_reason_blank'

            curr_start_seconds = curr_start_time/sample_rate
            if blank and len(sounds_in_this_current_window) > 0: #if blank
                for selection in sounds_in_this_current_window:
                    if selection['found']=='notfound':
                        selection['found']='blank'
                        selection['reason']=reason

            elif len(sounds_in_this_current_window) == 0 and not blank:#if no sounds in window, and model predicted something
                for tensor, percent in tensor_percents:
                    if not detection_values.get(tensor):
                        scores['false_positives'][tensor] = scores['false_positives'].get(tensor,0)+1
                
            else:
                for selection in sounds_in_this_current_window:
                    for tensor, percent in tensor_percents:
                        if selection['Code'] == tensor:
                            selection['found']='found'

            pop_list = [key for key in detection_values.keys()]
            for tensor, percent in tensor_percents:
                if detection_values.get(tensor):
                    dict_to_update = detection_values[tensor]
                    dict_to_update['End Time (s)'] = curr_start_seconds+2.7
                    dict_to_update['End File Samp (samples)'] = curr_start_time+2.7*sample_rate, #only update end times if item is already in
                    detection_values[tensor] = dict_to_update
                    pop_list.remove(tensor)
                
                else:
                    row = {
                    'Begin Path': sound,
                    'End File': sound.split('/')[-1],
                    'Begin Time (s)': curr_start_seconds,
                    'End Time (s)': curr_start_seconds+2.7,
                    'Beg File Samp (samples)': curr_start_time,
                    'End File Samp (samples)': curr_start_time+2.7*sample_rate,
                    'Code': tensor
                    }
                    detection_values[tensor] = row
            scores['items_predicted'] += len(pop_list)
            for code in pop_list:
                row = detection_values.pop(code)
                update_table.append(row)
            if end+detection_dur >= len_of_track - dur:
                scores['items_predicted'] += len(detection_values)
                for row in detection_values.values():
                    update_table.append(row)
            

    if model:
        del model
        gc.collect()
        return scores, update_table
    else:
        return scores, None


def run_through_audio(model_path, dict_path, vad_threshold, percentage_threshold,topk):
    #how many false alarms of sounds where there isnt
    #time taken (automatically done by wandb)
    #how many different sounds correctly picked up
        #how many were misclassified (as sounds)
    #how many sounds picked up (just add up all sound predictions that are actually sounds)
    #how many sounds misclassified as BLANK
    
    index_dict = {}
    model = None
    scores=None
    if model_path:
        with open(dict_path,'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                index_dict[int(row['Code'])] = row['Label']
        model = load_model_for_training(model_path,len(index_dict)) #assigning the correct model...
        # wandb.init(project='Testing Performance',name=model_path.split('/')[-1][:-3], entity="kasmello")
        scores = {
            'table_sounds': 0,
            'items_predicted': 0,
            'blank_reason_vad': 0,
            'blank_reason_blank': 0,
            'blank_reason_below_thresh': 0,
            'false_positives': {},
            'sounds_classified': {},
            'sounds_misclassified': {},
            'sounds_misclassified_blank': {}
        }
    all_tables = load_all_selection_tables()
    try:
        for table in all_tables:
            sound = load_all_sounds(table)
            dict_list = read_in_csv_times(table)
            scores, table_dict = process_and_predict(sound, scores, dict_list, model, index_dict, vad_threshold, percentage_threshold,topk)
            if model_path and not wandb.run:
                today = date.today()
                save_string = f"Selection Tables/{sound.split('/')[-1][:-4]}_{today}.txt"
                print(f'SAVING AS {save_string}')
                save_file(table_dict,save_string)
        number_false_positives = sum(scores['false_positives'].values())
            #number classified correctly: correct/table
        number_classified = sum(scores['sounds_classified'].values())
        #number classified incorrectly: incorrect/table
        number_misclassified = sum(scores['sounds_misclassified'].values())
        #number items classified as blank: blanks/items in table
        number_misclassified_blank = sum(scores['sounds_misclassified_blank'].values())
        try:
            ratio_items_predicted = round(scores['items_predicted']/scores['table_sounds'],3)
            classified = round(number_classified/scores['table_sounds'],3)
            misclassified = round(number_misclassified/scores['table_sounds'],3)
            misclassified_blank = round(number_misclassified_blank/scores['table_sounds'],3)
        except ZeroDivisionError:
            ratio_items_predicted = '∞'
            classified = '∞'
            misclassified = '∞'
            misclassified_blank = '∞'
            
        try:
            number_false_positives=round(number_false_positives/scores['items_predicted'],3)
        except ZeroDivisionError:
            number_false_positives = '∞'

        result_dict = {"Total Table Sounds": scores['table_sounds'], "Ratio Items Predicted": ratio_items_predicted,
            "Classified": classified, 
            "Misclassified": misclassified, 
            "Misclassified Blank": misclassified_blank,
            "Number False Positives": number_false_positives,
            "VAD Threshold": round(vad_threshold,3),
            "Percent Threshold": round(percentage_threshold,3),
            "Max Items": topk}
        print(result_dict)
        if wandb.run:
            wandb.log(result_dict)    
    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
    print(scores)

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
        fieldnames = list(table_dict[0].keys())
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

    elif model_name.lower() == 'deit-small':
        return timm.create_model('deit_small_distilled_patch16_224',pretrained=True, num_classes=num_labels, in_chans=1).to(device)

    elif model_name.lower()[0:8]=='vit-base':
        return timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=num_labels, in_chans=1).to(device)

    elif model_name.lower()=='vit-large':
        return timm.create_model('vit_large_patch16_224',pretrained=True, num_classes=num_labels, in_chans=1).to(device)

    elif model_name.lower()=='efficientnet':
        return timm.create_model('efficientnet_b0',pretrained=True, num_classes=num_labels, in_chans=1).to(device)

    elif model_name.lower()=='efficientnet5':
        return timm.create_model('efficientnet_b5',pretrained=True, num_classes=num_labels, in_chans=1).to(device)