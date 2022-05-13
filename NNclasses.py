'''
this is where I will write all my classes :)
'''

import os
import torch
import torchvision
import glob
import random
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from torchvision import transforms, datasets
from PIL import Image, ImageDraw
from scipy import signal
from scipy import ndimage
from scipy.io import wavfile
from pydub import AudioSegment

def wav_to_spectogram(item, save = True):
    fs, x = wavfile.read(item)
    Pxx, freqs, bins, im = plt.specgram(x, Fs=fs,NFFT=1024)
    # plt.pcolormesh(bins, freqs, 10*np.log10(Pxx))
    plt.imshow(10*np.log10(Pxx), cmap='gray_r')
    plt.axis('off')
    plt.show()
    if save:
        plt.savefig(f'{item[:-4]}.png',bbox_inches=0)
        print('saved!')

class nn_data:

    def __init__(self, root, batch_size):
        self.all_labels = self.make_folders(root)
        self.all_labels.sort()
        self.train_path, self.validation_path = self.stratify_sample(root)
        self.all_training, self.all_validation, self.label_dict = self.grab_dataset(batch_size)
        self.label_dict = {v: k for k, v in self.label_dict.items()}


    def inverse_encode(self, llist):
        return [self.label_dict[int(x.item())] for x in llist]


    def grab_dataset(self, batch_size):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.RandAugment(),
            transforms.ToTensor()
        ])
        train_folder = datasets.ImageFolder(self.train_path,transform=transform)
        all_training = torch.utils.data.DataLoader(train_folder,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=3)

        validation_folder = datasets.ImageFolder(self.validation_path,transform=transform)
        all_validation = torch.utils.data.DataLoader(validation_folder,
                                              batch_size=50,
                                              shuffle=True,
                                              num_workers=3)
        return all_training, all_validation, train_folder.class_to_idx


    def pad_all_spectograms(self,pad_ms=2200):
        for item in self.wavs:
            audio = AudioSegment.from_wav(item)
            if len(audio) != 2200:
                print(f'original length: {len(audio)}')
                silence1 = AudioSegment.silent(duration=int((pad_ms-len(audio))/2))
                silence2 = AudioSegment.silent(duration=ceil((pad_ms-len(audio))/2))
                new_length = len(audio)+len(silence1)+len(silence2)
                print(f'new length: {new_length}')
                padded = silence1 + audio + silence2  # Adding silence after the audio
                padded.export(item, format='wav')

    def make_folders(self, root):
        all_dir = [dir for root, dir, file in os.walk(root)]

        all_dir = all_dir[0]
        try:
            os.mkdir(os.path.join(root,'Training'))
        except FileExistsError:
            all_dir.remove('Training')
        try:
            os.mkdir(os.path.join(root,'Validation'))
        except FileExistsError:
            all_dir.remove('Validation')
        print(all_dir)
        return all_dir

    def stratify_sample(self, root):
        random.seed(30)
        for dir in self.all_labels:
            try:
                os.mkdir(os.path.join(root,'Training',dir))
            except FileExistsError:
                pass
            try:
                os.mkdir(os.path.join(root,'Validation',dir))
            except FileExistsError:
                pass
            label_dir = os.path.join(root,dir)
            all_files = [file for root, dir, file in os.walk(label_dir)]
            all_files = all_files[0]
            all_files = [file for file in all_files if file[-3:]=='png']
            random.seed(30)
            samp_length = int(len(all_files)*0.25)
            samp = random.sample(all_files,samp_length)
            if len(all_files)!= 0:
                for file in tqdm(all_files):
                    if file in samp:
                        old_dir = os.path.join(root,dir,file)
                        new_dir = os.path.join(root,'Validation',dir,file)
                        shutil.move(old_dir,new_dir)
                    else:
                        old_dir = os.path.join(root,dir,file)
                        new_dir = os.path.join(root,'Training',dir,file)
                        shutil.move(old_dir,new_dir)
        return os.path.join(root, 'Training'), os.path.join(root, 'Validation')

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        temp = torch.randn(224,224).view(-1,1,224,224)
        self._to_linear = None
        self.convs(temp)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,23)

    def __str__(self):
        return 'CNNet2'

    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)) #3 by 3 pooling
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

        if not self._to_linear:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(224*224,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,23)

    def __str__(self):
        return 'Net'

    def forward(self,x):
        x = F.relu(self.fc1(x)) #F.relu is an activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)#probability distribution

