import torch
import wandb
import ssl
import pathlib
import numpy as np
import csv
import math
import timm
import platform
from time import time
from tqdm import tqdm
from scipy.io import wavfile
from collections import deque
from skimage.transform import resize
from pattern_recognition import pattern_analyis
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.optim.lr_scheduler as lsr
from pydub import AudioSegment 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from NNclasses import Net, CNNet

ssl._create_default_https_context = ssl._create_unverified_context


def extract_f1_score(DATA, dict):
    data = [[category, dict[category]['f1-score']] for index, category in DATA.label_dict.items()]
    table = wandb.Table(data=data, columns = ["Label", "F1-score"])
    wandb.log({"F1-chart" : wandb.plot.bar(table, "Label", "F1-score",
                                title="F1-chart")})

def log_images(images,pred,actual):
    images = [wandb.Image(image) for image in images]
    data = [[pred[i],actual[i],images[i]] for i in range(len(images))]
    table = wandb.Table(data=data, columns = ["Predicted", "Actual",'Spectogram'])
    wandb.log({'Image Table': table})


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
        row['Begin Time (s)'] = float(row['Begin Time (s)']) + tocut;
    row['End Time (s)'] = float(row['Begin Time (s)']) + segment_dur
    return row
        

def read_in_csv_times(file):
    with open(file, 'r') as csv_file:
        csv_dict = csv.DictReader(csv_file,delimiter=',')
        for d in csv_dict:
            d['Selection'] = d.pop('\ufeffSelection')
        dict_list = []
        for i, row in enumerate(csv_dict):
            dict_list.append(pad_out_row(row, 3.2))
        return dict_list

def train_nn(DATA, lr=0.001, optimizer=optim.AdamW, net=None, epochs=5, lbl='',
             loss_f=F.nll_loss, momentum=None, wd=0,lr_decay=None):
    name = str(net)
    if len(str(net)) > 10:
        name = lbl
    if momentum:
        optimizer = optimizer(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        optimizer = optimizer(net.parameters(), lr=lr,weight_decay=wd)  # adamw algorithm
    scheduler = lr_decay(optimizer,1)
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave=False):
            net.train()
            x, y = batch
            net.zero_grad()
            # sets gradients at 0 after each batch
            output = F.log_softmax(net(x), dim=1)
            # calculate how wrong we are
            loss = loss_f(output, y)
            loss.backward()  # backward propagation
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
        net.eval()
        scheduler.step()
        final_layer = epoch == epochs - 1
        check_training_accuracy(DATA, net)
        validate_model(DATA, net, loss, final_layer)
    torch.save(net.state_dict(), f'{name}.nn')


def predict(data,net, num_batches=999999999):
    with torch.no_grad():
        pred = []
        actual = []
        for i, batch_v in enumerate(data):
            predicted = []
            label = []
            x, y = batch_v
            output = F.log_softmax(net(x), dim=1)
            for idx, e in enumerate(output):
                predicted.append(torch.argmax(e))
                label.append(y[idx])
            pred.extend(predicted)
            actual.extend(label)
            if i == num_batches-1 or i == len(data)-1:
                return x, predicted, label, pred, actual
        

def check_training_accuracy(DATA,net):
    images, predicted, label, pred, actual = predict(DATA.all_training, net, 125)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    print('\n\n')
    output = classification_report(actual, pred, output_dict=True)
    accuracy = output['accuracy']
    precision = output['weighted avg']['precision']
    recall = output['weighted avg']['recall']
    result_dict = {'T Accuracy': accuracy,
            'T Wgt Precision': precision, 'T Wgt Recall': recall}
    print(result_dict)
    wandb.log(result_dict)


def validate_model(DATA, net, loss, final_layer):
    images, predicted, label, pred, actual = predict(DATA.all_validation,net)
    predicted = DATA.inverse_encode(predicted)
    label = DATA.inverse_encode(label)
    pred = DATA.inverse_encode(pred)
    actual = DATA.inverse_encode(actual)
    print('\n\n')
    output = classification_report(actual, pred, output_dict=True)
    accuracy = output['accuracy']
    precision = output['weighted avg']['precision']
    recall = output['weighted avg']['recall']
    result_dict = {'Loss': loss, 'V Accuracy': accuracy,
            'V Wgt Precision': precision, 'V Wgt Recall': recall}
    print(result_dict)
    wandb.log(result_dict)
    if final_layer:
        log_images(images,predicted,label)
        extract_f1_score(DATA, output)
        print(classification_report(actual, pred))
        cm = confusion_matrix(actual,pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

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

    return transforms.ToTensor()(log_Pxx)

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
    if model_name=='Net':
        return Net()
    elif model_name=='CNNet2':
        return CNNet()
    elif model_name=='resnet18':
        return models.resnet18(pretrained=True)
    elif model_name=='vgg16':
        return models.vgg16(pretrained=True)
    elif model_name=='vit':
        return timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=23, in_chans=1)

def run_through_audio(model_path, dict_path):
    model_name = model_path.split('/')[-1][:-3]
    model = get_model_from_name(model_name)
    model.load_state_dict(torch.load(model_path))
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
                output = F.log_softmax(model(array_to_predict.float()),dim=1)
                prediction = index_dict[torch.argmax(output).item()]
                print(f'PREDICTION: {prediction}')
                pa.add_prediction(prediction, start, end, file, sound)
                temp_file.unlink()
        

def run_model(DATA,net,lr,wd,epochs,momentum, optimm='sgd', lr_decay=None):
    if optimm == 'sgd':
        optimm=optim.SGD

    elif optimm == 'adamw':
        optimm = optim.AdamW
        momentum=None

    device = torch.device("dml" if platform.system()=='Windows'
                                else "cpu")
    if lr_decay=='cosineAN':
        lr_decay = lsr.CosineAnnealingLR

    if net == 'net':
        model = Net()
        model = model.to(device)

        train_nn(DATA=DATA, net=model, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, momentum=momentum,lr_decay=lr_decay)
    elif net == 'cnnet':
        model = CNNet()
        model = model.to(device)
        train_nn(DATA=DATA, net=model, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, momentum=momentum,lr_decay=lr_decay)

    elif net == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 23)
        model = model.to(device)
        train_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='ResNet18',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)
    elif net == 'vgg16':
        model = models.vgg16(pretrained=True)
        first_conv_layer = [nn.Conv2d(
            1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)
        model.classifier[6].out_features = 23
        model = model.to(device)
        train_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='VGG16',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)
    elif net == 'vit':
        model = timm.create_model('vit_base_patch16_224',pretrained=True, num_classes=23, in_chans=1)
        model = model.to(device)
        train_nn(DATA=DATA, lr=lr, wd=wd, epochs=epochs, optimizer=optimm, net=model, lbl='ViT',
                                            loss_f=F.nll_loss, momentum=momentum,lr_decay=lr_decay)



