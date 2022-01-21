'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''


import os
import tkinter as tk
from tkinter import filedialog
from NNclasses import nn_label




if __name__ == '__main__':
    option = input('Hello. What would you like to do?\
                \n1: Select folders (Folder names as labels)\n')

    if option == '1':
        print('Please select the folder with the trained sounds')

        ## This code here opens a file selection dialog
        root = tk.Tk()
        file_path = filedialog.askdirectory()
        ##

        all_labels_str = [folder for folder in os.listdir(file_path) if \
        os.path.isdir(os.path.join(file_path,folder))] #list comprehension to add folders only

        all_training_labels = []
        for label in all_labels_str:
            label_class = nn_label(file_path,label)
            all_training_labels.append(label_class)

        for item in all_training_labels:
            print(item.label)

        
