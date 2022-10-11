'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import pathlib
import requests
import platform
from datetime import datetime
from NNfunctions import *
from NNtrainingfunctions import *
from NNclasses import nn_data

DATA = None
MODEL_PATH = None
LABEL_DICT_PATH = None


def find_root():
    """
    simple function to return string of data folder
    """
    if platform.system() == 'Darwin':  # MAC
        return '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback'
    elif platform.system() == 'Windows':
        return 'C://Users/Karmel 0481098535/Desktop/Humpback'

ROOT = find_root()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    try:
        requests.get('https://www.google.com')
    except requests.ConnectionError:
        print('Cannot connect to the internet, disabling online mode for WANDB')
        os.environ["WANDB_MODE"]='dryrun'
    wandb.init(project=name, name=name_str, entity="kasmello")
    run_model(DATA,name,lr,wd, epochs,momentum, optimm, lr_decay)

def find_file(extension, custom_string):
    """
    find file with specific extension
    input:
        extension - string of extension (e.g. csv)
        custom_string - string to print if no files with the extension are found
    output:
        str of file path
    """
    all_models = sorted(pathlib.Path.cwd().glob(f'*.{extension}'))
    ask_list = []
    for index, model_path in enumerate(all_models):
        model_path_str = model_path.as_posix()
        name = model_path_str.split('/')[-1]
        ask_list.append([f'{index}: {name}',model_path_str])
    if len(ask_list) == 0: 
        print(custom_string)
        raise FileNotFoundError(custom_string)
    ask_str = ''
    for item in ask_list:
        ask_str += item[0]+'\n'
    index_for_model = input(f'Type number to select a file: 0 to {len(ask_list)-1}\n{ask_str}')
    return ask_list[int(index_for_model)][1]


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
                    \n9: Train and Test VGG16\
                    \n10: Load model and tensor-to-label encoding\
                    \n11: Test loaded model on Validation Data\n')

        if option == '1':
            DATA = nn_data(ROOT, batch_size=32)

        elif option == '2':
            #use https://arxiv.org/pdf/2106.10270.pdf as reference
            lr=0.0007
            wd=0.03
            epochs=6
            momentum=0.9
            name='vit'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '3':
            DATA.test_transform()

        elif option == '4':
            try:
                MODEL_PATH = find_file('nn', 'You have no models constructed - make some models before doing this')
                LABEL_DICT_PATH = find_file('csv', 'You have no label dictionaries - make some before doing this')
            except FileNotFoundError as e:
                print(e)
            run_through_audio(MODEL_PATH, LABEL_DICT_PATH)

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
            lr_decay = None
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '7':
            lr=0.01
            wd=0.03
            epochs=10
            momentum=0.9
            name='cnnet'
            optimm='sgd'
            lr_decay = None
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '8':
            lr=0.001
            wd=0.03
            epochs=10
            momentum=0.9
            name='resnet18'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '9':
            lr=0.001
            wd=0.03
            epochs=10
            momentum=0.9
            name='vgg16'
            optimm='sgd'
            lr_decay = 'cosineAN'
            start_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '10':
            try:
                MODEL_PATH = find_file('nn', 'You have no models constructed - make some models before doing this')
                LABEL_DICT_PATH = find_file('csv', 'You have no label dictionaries - make some before doing this')
            except FileNotFoundError as e:
                print(e)

        elif option == '11':
            model = load_model_for_training(MODEL_PATH)
            validate_model(DATA,model,None,False)
