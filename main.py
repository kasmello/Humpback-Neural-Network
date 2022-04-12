'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''


import os
import torch
import progressbar
import numpy as np
import tkinter as tk
import tkmacosx as tkm
from sklearn import preprocessing
from sklearn import preprocessing
from tkinter import filedialog
from NNclasses import nn_label, wav_to_spectogram
from torchvision import datasets, transforms
from PIL import Image
from pydub import AudioSegment





if __name__ == '__main__':
    finished = False
    all_training_labels = []
    all_validation_labels = []
    while not finished:
        option = input('Hello. What would you like to do?\
                    \n1: Select folders (Folder names as labels)\
                    \n2: Test Vision Transformer model\
                    \n3: Generate spectograms\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Train Vision Transformer\
                    \n6: Train ResNet18 NN\
                    \n\t6b: Train ResNet18 NN (using csv matrix)\n')


        if option == '1':
            print('Please select the folder with the trained sounds')
            file_path = ''
            # This code here opens a file selection dialog
            # try:
            #     root = tk.Tk()
            #     file_path = filedialog.askdirectory()
            # except FileNotFoundError:
            #     print('Tkinter is extremely crinj')
            training_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/Training'
            validation_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/Validation'

            ##

            all_labels_str = [folder for folder in os.listdir(training_file_path) if \
            os.path.isdir(os.path.join(training_file_path,folder))] #list comprehension to add folders only
            all_val_labels_str = [folder for folder in os.listdir(validation_file_path) if \
            os.path.isdir(os.path.join(validation_file_path,folder))] #list comprehension to add folders only
            bar = progressbar.ProgressBar(maxval=len(all_labels_str)+len(all_val_labels_str), \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            i = 0

            for label in all_labels_str:
                label_class = nn_label(training_file_path,label)
                label_class.grab_all_spectograms()
                label_class.grab_all_csvs()
                all_training_labels.append(label_class)
                bar.update(i + 1)
                i += 1

            for label in all_val_labels_str:
                label_class = nn_label(validation_file_path,label)
                label_class.grab_all_spectograms()
                label_class.grab_all_csvs()
                all_validation_labels.append(label_class)
                bar.update(i + 1)
                i += 1
            bar.finish()


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
            #training resnet model
            train = []
            train_labels = []
            labels = []
            labels_dict = {}

            if len(option) == 2:
                for item in all_training_labels:
                    labels.append(item.labels[:-1])
                    train.extend(item.csvs)

                    for i in range(len(item.csvs)):
                        train_labels.append(item.labels[:-1])

            else:
                for item in all_training_labels:
                    labels.append(item.label[:-1])
                    train.extend(item.pngs)

                    for i in range(len(item.pngs)):
                        train_labels.append(item.label[:-1])
            train = torch.stack(train)

            le = preprocessing.LabelEncoder()
            le_labels = le.fit_transform(labels)
            for i in range(len(labels)):
                labels_dict[labels[i]] = le_labels[i]
            for i in range(len(train_labels)):
                train_labels[i] = labels_dict[train_labels[i]]
            train_labels = transforms.tensor(train_labels)
            train_data_and_labels = [train,train_labels]
            print(train_data_and_labels)
