'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''


import os
import tkinter as tk
import tkmacosx as tkm
from tkinter import filedialog
from NNclasses import nn_label
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np





if __name__ == '__main__':
    option = input('Hello. What would you like to do?\
                \n1: Select folders (Folder names as labels)\
                \n2: Test Vision Transformer model\
                \n3: Generate spectograms\
                \n4: Go through the Entire Dataset (BETA)')

    if option == '1':
        print('Please select the folder with the trained sounds')
        file_path = ''
        # This code here opens a file selection dialog
        # try:
        #     root = tk.Tk()
        #     file_path = filedialog.askdirectory()
        # except FileNotFoundError:
        #     print('Tkinter is extremely crinj')
        file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/All/'

        ##

        all_labels_str = [folder for folder in os.listdir(file_path) if \
        os.path.isdir(os.path.join(file_path,folder))] #list comprehension to add folders only

        all_training_labels = []
        for label in all_labels_str:
            label_class = nn_label(file_path,label)
            label_class.grab_all_spectograms()
            all_training_labels.append(label_class)

        for item in all_training_labels:
            print(item.label)
            print(item.path)
            print(item.pngs)

    elif option == '2':
        k = 10
        imagenet_labels = dict(enumerate(open('classes.txt')))

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
            label_class.convert_wavs_to_spectogram()
            all_training_labels.append(label_class)
            print('bit by bit...')
        print('Done!')
