'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import random
import platform
import numpy as np
from NNfunctions import *

DATA = None
if __name__ == '__main__':
    finished = False
    while not finished:
        option = input('\nHello. What would you like to do?\
                    \n1: Make Training and Validation Data\
                    \n2: Train and Test Vision Transformer model\
                    \n3: Generate spectograms\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Idk\
                    \n6: Train and Test NN\
                    \n7: Train and Test CNN\
                    \n8: Train and Test Pretrained ResNet-18\
                    \n9 Train and Test VGG16\n')

        if option == '1':
            if platform.system() == 'Darwin':  # MAC
                root = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
            elif platform.system() == 'Windows':
                root = 'C://Users/Karmel 0481098535/Desktop/Humpback'
            DATA = nn_data(root, batch_size=16)
            breakpoint()

        elif option == '2':
            lr=0.01
            weight_decay=0
            epochs=5
            name='vit'
            wandb.init(project=name, name=f'{datetime.now().strftime("%D %H:%M")} lr={lr} wd={weight_decay}', entity="kasmello")
            run_model(DATA,name,lr,weight_decay, epochs)

        elif option == '3':
            # print('Please select the folder you want to generate the spectograms from')
            # file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/All/'
            # print(f'Grabbing files from {file_path}')
            # all_labels_str = [folder for folder in os.listdir(file_path) if
            #                   os.path.isdir(os.path.join(file_path, folder))]  # list comprehension to add folders only
            #
            # all_training_labels = []
            # for label in all_labels_str:
            #     label_class = nn_label(file_path, label)
            #     label_class.save_spectogram()
            #     all_training_labels.append(label_class)
            #     print('bit by bit...')
            # print('Done!')
            pass
        elif option == '4':
            # theoryyy
            example_file_path = '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/braindead.wav'
            example_track = AudioSegment.from_wav(example_file_path)
            len_of_database = 10000
            size_of_chunks = 2  # in seconds
            for t in range(len_of_database - size_of_chunks + 1):
                start = t * 1000
                end = (t + size_of_chunks) * 1000
                print(f'START: {start} END: {end}')
                segment = example_track[start:end]
                segment.export(
                    '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav', format='wav')
                wav_to_spectogram(
                    '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav', save=False)
                os.remove(
                    '/Volumes/Macintosh HD/Users/karmel/Desktop/Results/_temp.wav')

        elif option == '5':

            # Training
            # vision_transformer = NNclasses.VisionTransformer()
            # for epoch in range(2):
            #     for item in all_training_labels:
            #         label = item.label
            #         for png in item.pngs:
            #             input = png
            pass

        elif option[0] == '6':
            lr=0.001
            weight_decay=0
            epochs=5
            name='net'
            wandb.init(project=name, name=f'{datetime.now().strftime("%D %H:%M")} lr={lr} wd={weight_decay}', entity="kasmello")
            run_model(DATA,name,lr,weight_decay, epochs)

        elif option == '7':
            lr=0.001
            weight_decay=0
            epochs=5
            name='cnnet'
            wandb.init(project=name, name=f'{datetime.now().strftime("%D %H:%M")} lr={lr} wd={weight_decay}', entity="kasmello")
            run_model(DATA,name,lr,weight_decay, epochs)

        elif option == '8':
            lr=0.01
            weight_decay=0
            epochs=5
            name='resnet18'
            wandb.init(project=name, name=f'{datetime.now().strftime("%D %H:%M")} lr={lr} wd={weight_decay}', entity="kasmello")
            run_model(DATA,name,lr,weight_decay, epochs)

        elif option == '9':
            lr=0.001
            weight_decay=0
            epochs=5
            name='vgg16'
            wandb.init(project=name, name=f'{datetime.now().strftime("%D %H:%M")} lr={lr} wd={weight_decay}', entity="kasmello")
            run_model(DATA,name,lr,weight_decay, epochs)