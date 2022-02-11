'''
this is where I will write all my classes :)
'''

import os
import torch
import torchvision
from torchvision import datasets, transforms
import helper
from PIL import Image, ImageDraw

class nn_label:


    def __init__(self, path,folder):
        self.label = folder
        self.path = os.path.join(path,folder) + '/'
        self.wavs = self.grab_all_spectograms()


    def grab_all_spectograms(self):
        all_files = [wav for root, dir, wav in os.walk(self.path)]
        all_files = all_files[0]
        all_wavs = []
        for wav in all_files:
            if wav[-4:]=='.png':
                file_name = os.path.join(self.path,wav)
                im = Image.open(file_name)
                all_wavs.append(im)



        return all_wavs
