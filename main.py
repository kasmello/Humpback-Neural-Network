'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import cv2
import torch
import random
import NNclasses
import platform
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import classification_report
from NNclasses import nn_data, Net, CNNet
from NNfunctions import train_pretrained_nn, train_nn, validate_model


DATA = None

if __name__ == '__main__':
    finished = False
    while not finished:
        option = input('\nHello. What would you like to do?\
                    \n1: Make Training and Validation Data\
                    \n2: Test Vision Transformer model\
                    \n3: Generate spectograms\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Train Vision Transformer\
                    \n6: Train and Test NN\
                    \n7: Train and Test CNN\
                    \n8: Train and Test Pretrained ResNet-18\
                    \n9 Train and Test VGG16\n')

        if option == '1':
            if platform.system()=='Darwin':#MAC
                root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
            elif platform.system()=='Windows':
                root = 'C://Users/Karmel 0481098535/Desktop/Humpback'
            DATA = nn_data(root, batch_size = 16)
            breakpoint()

        elif option == '1b':
            with open('all_training.ml','rb') as file:
                all_training = pickle.load(file)
            with open('all_validation.ml','rb') as file:
                all_validation = pickle.load(file)

        elif option == '2':

            model = torch.load('model.pth')
            model.eval()

            cat = (np.array(Image.open('cat.png'))/128)-1
            inp = torch.from_numpy(cat).permute(2,0,1).unsqueeze(0).to(torch.float32)
            logits = model(inp)
            probs = torch.nn.functional.softmax(logits,dim=-1)

            top_probs, top_ics = probs[0].topk(k)
            # turn logits into probabilities

            print('\nPREDICTING IMAGE 1\n')
            for i, (ix_, prob_) in enumerate(zip(top_ics, top_probs)):
                ix = ix_.item()
                prob = prob_.item()
                cls = imagenet_labels[ix].strip()
                print(f"{i}: {cls:<45} --- {prob:.4f}")

        elif option == '3':
            print('Please select the folder you want to generate the spectograms from')
            file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/All/'
            print(f'Grabbing files from {file_path}')
            all_labels_str = [folder for folder in os.listdir(file_path) if \
            os.path.isdir(os.path.join(file_path,folder))] #list comprehension to add folders only

            all_training_labels = []
            for label in all_labels_str:
                label_class = nn_label(file_path,label)
                label_class.save_spectogram()
                all_training_labels.append(label_class)
                print('bit by bit...')
            print('Done!')
        elif option == '4':
            #theoryyy
            example_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/braindead.wav'
            example_track = AudioSegment.from_wav(example_file_path)
            len_of_database = 10000
            size_of_chunks = 2 # in seconds
            for t in range(len_of_database-size_of_chunks+1):
                start = t * 1000
                end = (t+size_of_chunks)*1000
                print(f'START: {start} END: {end}')
                segment = example_track[start:end]
                segment.export('/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav', format='wav')
                wav_to_spectogram('/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav',save=False)
                os.remove('/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav')

        elif option == '5':

            #Training
            vision_transformer = NNclasses.VisionTransformer()
            for epoch in range(2):
                for item in all_training_labels:
                    label = item.label
                    for png in item.pngs:
                        input = png

        elif option[0] == '6':
            actual, pred = train_nn(DATA=DATA,net = Net(), lr = 0.001)

        elif option == '7':
            actual, pred = train_nn(DATA=DATA,net = CNNet(), lr = 0.001)

        elif option == '8':
            model = models.resnet18()
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
            model = model.to(device)
            model.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3,bias=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 23)
            lr = 0.01
            momentum = 0.9
            epochs=10
            actual, pred = train_pretrained_nn(DATA=DATA,lr=lr,optimizer=optim.SGD,net=model,epochs=epochs,lbl='ResNet18',\
                            loss_f=F.nll_loss, momentum=momentum)


        elif option == '9':
            model = models.vgg16()
            lr = 0.001
            momentum = 0.9
            epochs=10
            first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
            first_conv_layer.extend(list(model.features))
            model.features= nn.Sequential(*first_conv_layer )
            model.classifier[6].out_features = 23
            actual, pred = train_pretrained_nn(DATA=DATA,lr=lr,optimizer=optim.SGD,net=model,epochs=epochs,lbl='VGG16',\
                            loss_f=F.nll_loss, momentum=momentum)
