'''
this is where the class containing tensor data/training functions is coded!
'''
import pathlib
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
from transformclasses import *
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1
class nn_data:

    def __init__(self, root, batch_size,pink=True, specgram=True):
        self.all_labels = nn_data.make_folders(root)
        self.all_labels.sort()
        self.batch_size=batch_size
        if not pink: 
            pink_p = 0
        else:
            pink_p = 0.7
        if not specgram:
            specgram_p = 0
        else:
            specgram_p = 0.35
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            TimeWarp(p=specgram_p,T=80),
            FreqMask(p=specgram_p, F=10),
            TimeMask(p=specgram_p, T=10),
            TranslateHorizontal(p=0.5,moving=60),
            AddPinkNoise(p=pink_p,power=1)
        ])
        
        self.transform_all = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            AddPinkNoise(p=1,power=1), 
            TimeWarp(p=1,T=100),
            FreqMask(p=1, F=20),
            TimeMask(p=1, T=20),
            TranslateHorizontal(p=1,moving=100),
            AddPinkNoise(p=pink_p,power=1)
        ])

        self.v_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.stratify_sample(root)
        self.grab_dataset(batch_size)
        self.save_label_dict()

    def save_label_dict(self):
        with open(f'index_to_label.csv','w') as file:
            file.write('Code,Label\n')
            for key, item in self.label_dict.items():
                file.write(f'{key},{item}\n')


    def inverse_encode(self, labels):
        if len(labels) == 1:
            return self.label_dict[int(labels[0].item())]
        return [self.label_dict[int(x.item())] for x in labels]


    def grab_dataset(self, batch_size):
        train_folder = datasets.ImageFolder(self.train_path,transform=self.train_transform)
        self.all_training = torch.utils.data.DataLoader(train_folder,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=np.where(platform.system()=="Windows",2,0),
                                              pin_memory=True)

        validation_folder = datasets.ImageFolder(self.validation_path,transform=self.v_transform)
        self.all_validation = torch.utils.data.DataLoader(validation_folder,
                                              batch_size=512,
                                              shuffle=True,
                                              num_workers=np.where(platform.system()=="Windows",2,0),
                                              pin_memory=True)
        testing_folder = datasets.ImageFolder(self.testing_path,transform=self.v_transform)
        self.all_testing = torch.utils.data.DataLoader(testing_folder,
                                              batch_size=512,
                                              shuffle=True,
                                              num_workers=np.where(platform.system()=="Windows",2,0),
                                              pin_memory=True)
        self.label_dict = {v: k for k, v in train_folder.class_to_idx.items()}

    def test_transform(self):
        categories_checked = ['LM','C','D']
        images = []
        for category in categories_checked:
            image_path = pathlib.Path(os.path.join(self.train_path, category)).glob('*.png')
            for path in image_path:
                images.append(path.as_posix())
                break
        for img_path in images:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()
            plt.close()
            # stat = ImageStat.Stat(img)
            example = self.transform_all(img)
            plt.imshow(example[0], cmap='gray')
            plt.show()
            plt.close()

    @staticmethod
    def make_folders(root):
        all_dir = [dir.name for dir in pathlib.Path(root).glob('*') if dir.name not in ['Training','Validation','Testing','.DS_Store']]
        try:
            pathlib.Path(os.path.join(root,'Training')).mkdir(parents=True, exist_ok=True)
        except FileExistsError: #this shouldnt happen with parents=true but it does
            pass
        try:
            pathlib.Path(os.path.join(root,'Validation')).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        try:
            pathlib.Path(os.path.join(root,'Testing')).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            pass
        print(all_dir)
        return all_dir

    def stratify_sample(self, root):
        random.seed(30)
        print('STRATIFYING')
        for dir in tqdm(self.all_labels):
            try:
                pathlib.Path(os.path.join(root,'Training',dir)).mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            try:
                pathlib.Path(os.path.join(root,'Validation',dir)).mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            try:
                pathlib.Path(os.path.join(root,'Testing',dir)).mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            label_dir = os.path.join(root,dir)
            all_files = [file for file in pathlib.Path(label_dir).glob('**/*.png')]
            random.seed(30)
            train, validate, test = np.split(np.arange(0,len(all_files)), [int(train_ratio*len(all_files)), 
                int((train_ratio+validation_ratio)*len(all_files))])
            if len(all_files)!= 0:
                for i in train:
                    old_dir = os.path.join(root,dir,all_files[i].name)
                    new_dir = os.path.join(root,'Training',dir,all_files[i].name)
                    shutil.move(old_dir,new_dir)
                for i in validate:
                    old_dir = os.path.join(root,dir,all_files[i].name)
                    new_dir = os.path.join(root,'Validation',dir,all_files[i].name)
                    shutil.move(old_dir,new_dir)
                for i in test:
                    old_dir = os.path.join(root,dir,all_files[i].name)
                    new_dir = os.path.join(root,'Testing',dir,all_files[i].name)
                    shutil.move(old_dir,new_dir)

        self.train_path = os.path.join(root,'Training')
        self.validation_path = os.path.join(root,'Validation')
        self.testing_path = os.path.join(root,'Testing')

class CNNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3 = nn.Conv2d(64,128,3)
        temp = torch.randn(224,224).view(-1,1,224,224)
        self._to_linear = None
        self.convs(temp)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,num_classes)

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
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(224*224,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,num_classes)

    def __str__(self):
        return 'Net'

    def forward(self,x):
        x = F.relu(self.fc1(x.view(-1, 224 * 224))) #F.relu is an activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def deit_base_patch16_224(num_classes, in_chans, pretrained=False):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
        model.num_classes=num_classes
        model.patch_embed.proj.in_channels = in_chans

    return model