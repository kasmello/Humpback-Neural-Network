'''
this is where the class containing tensor data/training functions is coded!
'''

import os
import torch
import random
import shutil
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform
from tqdm import tqdm
from math import ceil
from datetime import datetime
from torchvision import transforms, datasets
from PIL import Image, ImageStat
from pydub import AudioSegment
from transformclasses import FreqMask, TimeMask, TimeWarp
# from spec_augment_pytorch import spec_augment #give credits where
class nn_data:

    def __init__(self, root, batch_size):
        self.all_labels = nn_data.make_folders(root)
        self.all_labels.sort()
        self.stratify_sample(root)
        self.grab_dataset(batch_size)
        self.save_label_dict()

    def save_label_dict(self):
        curr_datetime = datetime.now().isoformat(timespec='hours')
        with open(f'{curr_datetime}_index_to_label.csv','w') as file:
            file.write('Code,Label\n')
            for key, item in self.label_dict.items():
                file.write(f'{key}, {item}\n')


    def inverse_encode(self, llist):
        return [self.label_dict[int(x.item())] for x in llist]


    def grab_dataset(self, batch_size):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            TimeWarp(p=0.2,T=50),
            FreqMask(p=0.2, F=20),
            TimeMask(p=0.2, T=20),
        ])

        v_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        train_folder = datasets.ImageFolder(self.train_path,transform=transform)
        self.all_training = torch.utils.data.DataLoader(train_folder,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=3)

        validation_folder = datasets.ImageFolder(self.validation_path,transform=v_transform)
        self.all_validation = torch.utils.data.DataLoader(validation_folder,
                                              batch_size=50,
                                              shuffle=True,
                                              num_workers=3)
        self.label_dict = {v: k for k, v in train_folder.class_to_idx.items()}

    def test_transform(self):
        img = Image.open(os.path.join(self.train_path, 'LM', 'LM-28.png'))
        stat = ImageStat.Stat(img)
        mean = stat.mean[:1]
        std = stat.stddev[:1]
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean = mean, std = std),
                TimeWarp(p=1, T=50),
                FreqMask(p=1, F=20),
                TimeMask(p=1, T=20),                
            ])
        
        print(img)
        example = transform(img)
        breakpoint()
        plt.imshow(example[0])
        plt.show()
        plt.close()

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

    @staticmethod
    def make_folders(root):
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
        self.train_path = os.path.join(root,'Training')
        self.validation_path = os.path.join(root,'Validation')

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
        self.fc2 = nn.Linear(512,25)

    def __str__(self):
        return 'CNNet'

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
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(224*224,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,25)

    def __str__(self):
        return 'Net'

    def forward(self,x):
        x = F.relu(self.fc1(x.view(-1, 224 * 224))) #F.relu is an activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

