'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import platform
from datetime import datetime
from NNfunctions import *
from NNclasses import nn_data

DATA = None

def start_model(name,lr,wd,momentum,epochs, optimm, lr_decay):
    name_str = f'{datetime.now().strftime("%D %H:%M")}'
    if lr:
        name_str += f' lr={lr}'
    if wd:
        name_str += f' wd={wd}'
    if momentum:
        name_str += f' momentum={momentum}'
    if optimm:
        name_str += f' optimm={optimm}'
    if lr_decay:
        name_str += f' lr_decay={lr_decay}'
    wandb.init(project=name, name=name_str, entity="kasmello")
    run_model(DATA,name,lr,wd, epochs,momentum, optimm, lr_decay)


if __name__ == '__main__':
    finished = False
    while not finished:
        option = input('\nHello. What would you like to do?\
                    \n1: Make Training and Validation Data\
                    \n2: Train and Test Vision Transformer model\
                    \n3: Test how the transform looks on example data\
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
            lr=0.0005
            wd=0.03
            epochs=5
            momentum=0.9
            name='vit'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '3':

            DATA.test_transform()
        elif option == '4':
            test_folder = '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/Testing'
            try:
                os.makedirs(test_folder)
            except FileExistsError:
                pass
            run_through_audio()

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
            wd=0.03
            epochs=5
            momentum=0.9
            name='net'
            optimm='adamw'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '7':
            lr=0.001
            wd=0.03
            epochs=5
            momentum=0.9
            name='cnnet'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '8':
            lr=0.0005
            wd=0.03
            epochs=5
            momentum=0.9
            name='resnet18'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '9':
            lr=0.0005
            wd=0.03
            epochs=5
            momentum=0.9
            name='vgg16'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)