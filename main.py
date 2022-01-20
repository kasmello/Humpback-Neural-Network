'''
This is the main module of the Neural Network Program to automatically
detect Humpback whales
'''


import os
import tkinter as tk
from tkinter import filedialog




if __name__ == '__main__':
    option = input('Hello. What would you like to do?\
                \n1: Select folders (Folder names as labels)\n')

    if option == '1':
        print('Please select the folder with the trained sounds')

        ## This code here opens a file selection dialog
        root = tk.Tk()
        file_path = filedialog.askdirectory()
        ##

        all_labels = [folder for folder in os.listdir(file_path) if \
        os.path.isdir(os.path.join(file_path,folder))] #list comprehension to add folders only
        print(all_labels)
