'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''


import os
import torch
import progressbar
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkmacosx as tkm
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import classification_report
from tkinter import filedialog
from NNclasses import nn_label, wav_to_spectogram, Net

def shuffle_list(data):
    index = list(range(0,len(data[0])))
    random.shuffle(index)
    shuffled_matrices = [data[0][x] for x in index]
    shuffled_index = [data[1][x] for x in index]
    return [shuffled_matrices,shuffled_index]

def make_sublist(data):
    chunk_length = int(len(data[0])/10)
    chunks_matrix = [torch.tensor(data[0][x:x+chunk_length]) for x in range(0, len(data[0]), chunk_length)]
    chunks_label = [torch.tensor(data[1][x:x+chunk_length]) for x in range(0, len(data[1]), chunk_length)]
    return [chunks_matrix,chunks_label]

def get_x_and_y(llist):
    x = []
    y = []
    for item in llist:
        x.extend(item.csvs)

        for i in range(len(item.csvs)):
            y.append(item.label)

    return x,y

def make_x_and_y_set(x,y, le):
    y = le.transform(y)
    x_and_y = [x,y]
    x_and_y = shuffle_list(x_and_y)
    x_and_y = make_sublist(x_and_y)
    return x_and_y

if __name__ == '__main__':
    finished = False
    all_training_labels = []
    all_validation_labels = []
    while not finished:
        option = input('Hello. What would you like to do?\
                    \n1: Select folders (Folder names as labels)\
                    \n\t1b: Load pickled labels\
                    \n2: Test Vision Transformer model\
                    \n3: Generate spectograms\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Train Vision Transformer\
                    \n6: Train and Test NN\
                    \n7: Test a General Neural Network\n')


        if option == '1':
            print('Please select the folder with the trained sounds')
            file_path = ''
            # This code here opens a file selection dialog
            # try:
            #     root = tk.Tk()
            #     file_path = filedialog.askdirectory()
            training_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/Training'
            validation_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/Validation'
            all_labels_str = [folder for folder in os.listdir(training_file_path) if \
            os.path.isdir(os.path.join(training_file_path,folder))] #list comprehension to add folders only
            all_val_labels_str = [folder for folder in os.listdir(validation_file_path) if \
            os.path.isdir(os.path.join(validation_file_path,folder))] #list comprehension to add folders only
            bar = progressbar.ProgressBar(maxval=len(all_labels_str)+len(all_val_labels_str)+2, \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            i = 0

            for label in all_labels_str:
                label_class = nn_label(training_file_path,label)
                label_class.grab_all_csvs()
                all_training_labels.append(label_class)
                bar.update(i + 1)
                i += 1

            for label in all_val_labels_str:
                label_class = nn_label(validation_file_path,label)
                label_class.grab_all_csvs()
                all_validation_labels.append(label_class)
                bar.update(i + 1)
                i += 1

            with open('all_training_labels.ml','wb') as file:
                pickle.dump(all_training_labels,file)
            bar.update(i+1)
            i += 1
            with open('all_validation_labels.ml','wb') as file:
                pickle.dump(all_validation_labels,file)
            bar.update(i+1)
            bar.finish()

        elif option == '1b':
            with open('all_training_labels.ml','rb') as file:
                all_training_labels = pickle.load(file)
            with open('all_validation_labels.ml','rb') as file:
                all_validation_labels = pickle.load(file)

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
            labels = []

            for item in all_training_labels:
                labels.append(item.label)

            train, train_labels = get_x_and_y(all_training_labels)
            validation, validation_labels = get_x_and_y(all_validation_labels)
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            train_data_and_labels = make_x_and_y_set(train,train_labels, le)
            validation_data_and_labels = make_x_and_y_set(validation,validation_labels, le)
            print(train_data_and_labels)
            print('\n\n==================Tensoring was a success!==================\n\n')
            net = Net()
            optimizer = optim.Adam(net.parameters(),lr = 0.0005) #adam algorithm
            epochs = 10

            for epoch in range(epochs):
                x,y = train_data_and_labels[0],train_data_and_labels[1]

                for i in range(len(x)):
                    net.zero_grad()
                     #sets gradients at 0 after each batch
                    output = net(x[i].view(-1,220*220))
                    #calculate how wrong we are
                    loss = F.nll_loss(output,y[i])
                    loss.backward()#backward propagation
                    optimizer.step()
                print(loss)

            results = np.zeros(len(labels)**2)
            results = results.reshape(len(labels),len(labels))
            pred = []
            actual = []

            with torch.no_grad():
                x,y = validation_data_and_labels[0],validation_data_and_labels[1]

                for i in range(len(x)):
                    output = net(x[i].view(-1,220*220))

                    for idx, e in enumerate(output):
                        pred.append(le.inverse_transform([torch.argmax(e)]))
                        actual.append(le.inverse_transform([y[i][idx]]))
            print('\n\n')
            print(classification_report(actual,pred))
            print('\n\n')

        elif option == '7':
            test = Net()
            print('\nTesting Creation of Neural Network:\n\n')
            x = torch.rand((220,220))
            x = x.view(1,220*220)
            test(x)
            print('Works!\n')
