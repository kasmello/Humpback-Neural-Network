# Humpback-Neural-Network

## Background

This program was used to write my dissertation for my final Undergraduate Degree in Data Science. It applies deep neural networks to Humpback Whale spectrograms and other aquatic sounds.

## Module guide

This is a short guide on what each module in this folder does. It is not yet optimised to use on any computer without changing many path directories, so please refer to below on what each notable module does.

- main.py - contains UI for the program, allowing users to train and load models, check out the effects of different augmentations, etc. To change parameters, one would have to go into the module to change the values manually.

- NNclasses.py - Class for loading the data, as well as some scrapped Neural Networks.

- NNfunctions.py - This module contains functions commonly used, with most being used for extracting the spectrograms out of the audio files. An important function from this module is *get_model_from_name*, which fetches the pre-trained model based on a string input and returns it.

- NNtrainingfunctions.py - This module contains all the functions used for training, with the exception of augmentation. This includes functions which propagate the data forward into the networks, measuring the metrics, and logging training/validation/testing data.

- read_audio_module.py - Can convert WAV files into spectrograms containing 2.7s of Audio Segments. One would also require the selection table containing the locations of all the sounds in the WAV file to do this. This is ran before training to prepare the data.

- transformclasses.py - Module containing augmentation classes and functions

- switch_mac_pc.py - Module which easily changed the path of files inside the Raven Pro selection tables from Macos to PC and vice-versa.

- unsupervised_vad.py - Voice activity detection functions, adapted from https://github.com/idnavid/py_vad_tool/blob/master/unsupervised_vad.py to fit this project. 

- generate_all_sweeps.sh - Bash script which perpares a sweeping project to take place. Produces a .txt file which is important for starting the sweep

- run_all_sweeps.sh - runs the sweep using the .txt file and sweep.py

- sweep.py - used by run_all_sweeps.sh to begin grid search of selected neural networks and parameters (defined in the *yaml* files)


