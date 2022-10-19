'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''

import os
import pathlib
import platform
from datetime import datetime
from NNfunctions import *
from NNtrainingfunctions import *
from NNclasses import nn_data
from read_audio_module import grab_spectogram

DATA = None
MODEL_PATH = None
LABEL_DICT_PATH = 'index_to_label.csv'



def find_root():
    """
    simple function to return string of data folder
    """
    if platform.system() == 'Darwin':  # MAC
        return '/Volumes/Macintosh HD/Users/karmel/Desktop/Training/Humpback/clean'
    elif platform.system() == 'Windows':
        return 'C://Users/Karmel 0481098535/Desktop/Humpback/clean'

ROOT = find_root()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def find_file(path,search_string):
    """
    find file with specific extension
    input:
        extension - string of extension (e.g. csv)
    output:
        str of file path
    """
    all_files = sorted(pathlib.Path(path).glob(search_string))
    ask_list = []
    for index, file_path_obj in enumerate(all_files):
        file_path = file_path_obj.as_posix()
        name = file_path.split('/')[-1]
        ask_list.append([f'{index}: {name}',file_path])
    ask_str = ''
    for item in ask_list:
        ask_str += item[0]+'\n'
    index_for_model = input(f'Type number to select a file: 0 to {len(ask_list)-1}\n{ask_str}')
    return ask_list[int(index_for_model)][1]

def load_model_and_dict():
    try:
        if not os.path.exists('Models'):
            raise FileNotFoundError('You have no models!')
        if not os.path.exists(LABEL_DICT_PATH):
            raise FileNotFoundError('index to label csv not found!')
        model_chosen = find_file('Models','*')
        model_path = find_file(model_chosen,'*.nn')
        return model_path
    except FileNotFoundError as e:
        print(e)


if __name__ == '__main__':
    finished = False
    while not finished:
        option = input('\nHello. What would you like to do?\
                    \n0: Quit\
                    \n1: Make Training and Validation Data\
                    \n2: Train and Test Vision Transformer model\
                    \n3: Test how the transform looks on example data\
                    \n4: Go through the Entire Dataset (BETA)\
                    \n5: Calculate Energy Levels of spectogram\
                    \n6: Train and Test NN\
                    \n7: Train and Test CNN\
                    \n8: Train and Test Pretrained ResNet-18\
                    \n9: Train and Test VGG16\
                    \n10: Load model\
                    \n11: Test loaded model on Validation Data\
                    \n12: Waveform/Spectrogram playground\n')

        if option == '0':
            print('BYE')
            finished = True

        if option == '1':
            DATA = nn_data(ROOT, batch_size=16)

        elif option == '2':
            #use https://arxiv.org/pdf/2106.10270.pdf as reference
            lr=0.0003
            wd=0.03
            epochs=6
            momentum=0.9
            name='vit'
            optimm='sgd'
            lr_decay = None
            run_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '3':
            DATA.test_transform()

        elif option == '4':
            if not MODEL_PATH:
                MODEL_PATH = load_model_and_dict()
            run_through_audio(MODEL_PATH, LABEL_DICT_PATH)

        elif option == '5':

            get_psd_distribution_of_all_images(DATA)

        elif option[0] == '6':
            lr=0.001
            wd=0.03
            epochs=5
            momentum=0.9
            name='net'
            optimm='adamw'
            lr_decay = None
            run_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '7':
            lr=0.01
            wd=0.03
            epochs=10
            momentum=0.9
            name='cnnet'
            optimm='sgd'
            lr_decay = None
            run_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '8':
            lr=0.001
            wd=0.03
            epochs=10
            momentum=0.9
            name='resnet18'
            optimm='sgd'
            lr_decay = 'cosineAN'
            run_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '9':
            lr=0.001
            wd=0.03
            epochs=10
            momentum=0.9
            name='vgg16'
            optimm='sgd'
            lr_decay = 'cosineAN'
            run_model(name,lr,wd,momentum,epochs, optimm, lr_decay)

        elif option == '10':
            MODEL_PATH = load_model_and_dict()

        elif option == '11':
            model = load_model_for_training(MODEL_PATH, len(DATA.all_labels))
            validate_model(DATA,model,None,True)

        elif option == '12':
            wavform, clean, sample_rate = grab_spectogram('Test Wavs/20090617040001.wav')
            plt.plot(wavform[0][0:8000])
            plt.show()
