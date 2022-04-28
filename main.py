'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import cv2
import torch
import pickle
import random
import NNclasses
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm
from sklearn.metrics import classification_report
from NNclasses import nn_data, Net, CNNet


DATA = None

def train_pretrained_nn(lr=0.001,optimizer=optim.AdamW,net=None,epochs=10,lbl='',\
                loss_f=F.nll_loss, momentum=None):
    name = str(net)
    if len(str(net))>10:
        name = lbl
    wandb.init(project=name, name=f'lr={lr}',entity="kasmello")
    if momentum:
        optimizer = optimizer(net.parameters(),lr = lr,momentum = momentum)
    else:
        optimizer = optimizer(net.parameters(),lr = lr) #adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave = False):
            net.train()
            x,y = batch
            net.zero_grad()
             #sets gradients at 0 after each batch
            output = F.log_softmax(net(x),dim=1)
            #calculate how wrong we are
            loss = loss_f(output,y)
            loss.backward()#backward propagation
            optimizer.step()
        net.eval()
        actual, pred = validate_model(net,loss)
        if epoch == epochs-1:
            return actual, pred

def train_nn(lr=0.001,optimizer=optim.AdamW,net=None,epochs=10,lbl='',loss_f=F.nll_loss, momentum=None):
    name = str(net)
    if len(str(net))>10:
        name = lbl
    wandb.init(project=name, name=f'lr={lr}',entity="kasmello")
    if momentum:
        optimizer = optimizer(net.parameters(),lr = lr,momentum = momentum)
    else:
        optimizer = optimizer(net.parameters(),lr = lr) #adamw algorithm
    for epoch in tqdm(range(epochs)):
        for batch in tqdm(DATA.all_training, leave = False):
            x,y = batch
            net.zero_grad()
             #sets gradients at 0 after each batch
            if str(net)=='Net':
                output = net(x.view(-1,224*224))
            else:
                output = net(x)
            #calculate how wrong we are
            loss = loss_f(output,y)
            loss.backward()#backward propagation
            optimizer.step()
        actual, pred = validate_model(net,loss)
        if epoch == epochs-1:
            return actual, pred



def validate_model(net,loss):
    with torch.no_grad():
        pred = []
        actual = []

        for batch_v in DATA.all_validation:
            x,y = batch_v
            if str(net)=='Net':
                output = net(x.view(-1,224*224))
            else:
                output = net(x)

            for idx, e in enumerate(output):
                pred.append(torch.argmax(e))
                actual.append(y[idx])
        pred = DATA.inverse_encode(pred)
        actual = DATA.inverse_encode(actual)
        print('\n\n')
        output = classification_report(actual,pred, output_dict = True)
        accuracy = output['accuracy']
        precision = output['weighted avg']['precision']
        recall = output['weighted avg']['recall']
        print({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
        wandb.log({'Loss': loss, 'Validation Accuracy': accuracy, 'Wgt Precision': precision, 'Wgt Recall': recall})
        return actual,pred

if __name__ == '__main__':
    finished = False
    while not finished:
        option = input('\nHello. What would you like to do?\
                    \n1: Select folders (Folder names as labels)\
                    \n2: Test Vision Transformer model\
                    \n3: Generate spectograms\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Train Vision Transformer\
                    \n6: Train and Test NN\
                    \n7: Train and Test CNN\
                    \n8: Train and Test Pretrained ResNet-18\
                    \n9 Train and Test VGG16\n')

        if option == '1':
            # This code here opens a file selection dialog
            # try:
            #     root = tk.Tk()
            #     file_path = filedialog.askdirectory()
            root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
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
            # try:
            #     root = tk.Tk()
            #     file_path = filedialog.askdirectory()
            # except :
            #     print('Tkinter is extremely crinj')
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
            actual, pred = train_nn(net = Net(), lr = 0.001)
            print(classification_report(actual,pred))

        elif option == '7':
            actual, pred = train_nn(net = CNNet(), lr = 0.001)
            print(classification_report(actual,pred))

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
            actual, pred = train_pretrained_nn(lr=lr,optimizer=optim.SGD,net=model,epochs=epochs,lbl='ResNet18',\
                            loss_f=F.nll_loss, momentum=momentum)
            print(classification_report(actual,pred))


        elif option == '9':
            model = models.vgg16()
            lr = 0.0007
            momentum = 0.9
            epochs=10
            first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
            first_conv_layer.extend(list(model.features))
            model.features= nn.Sequential(*first_conv_layer )
            model.classifier[6].out_features = 23
            actual, pred = train_pretrained_nn(lr=lr,optimizer=optim.SGD,net=model,epochs=epochs,lbl='VGG16',\
                            loss_f=F.nll_loss, momentum=momentum)
            print(classification_report(actual,pred))
