#! /usr/bin/python

# Voice Activity Detection (VAD) tool.
# use the vad_help() function for instructions.
# Navid Shokouhi December 2012.

# Updated: May 2017 for Speaker Recognition collaboration.

from audio_tools import *
import numpy as np

import matplotlib.pyplot as plt

##Function definitions:
def vad_help():
    """Voice Activity Detection (VAD) tool.
	
	Navid Shokouhi May 2017.
    """
    print("Usage:")
    print("python unsupervised_vad.py")

#### Display tools
def plot_this(s,title=''):
    """
     
    """
    
    s = s.squeeze()
    if s.ndim ==1:
        plt.plot(s)
    else:
        plt.imshow(s,aspect='auto')
        plt.title(title)
    plt.show()

def plot_these(s1,s2):
    try:
        # If values are numpy arrays
        plt.plot(s1/max(abs(s1)),color='red')
        plt.plot(s2/max(abs(s2)),color='blue')
    except:
        # Values are lists
        plt.plot(s1,color='red')
        plt.plot(s2,color='blue')
    plt.legend()
    plt.show()


#### Energy tools
def zero_mean(xframes):
    """
        remove mean of framed signal
        return zero-mean frames.
        """
    m = np.mean(xframes,axis=1)
    xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
    return xframes

def compute_nrg(xframes):
    # calculate per frame energy
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)
    #so nrg is sum of frequencies squared divided by frames 

def compute_log_nrg(xframes):
    # calculate per frame energy in log
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
    """
        x: input signal, each row is one frame
        """
    X = np.fft.fft(xframes,axis=1)
    X = np.abs(X[:,:X.shape[1]/2])**2
    return np.sqrt(X)



def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
    """
        Picks frames with high energy as determined by a 
        user defined threshold.
        
        This function also uses a 'context' parameter to
        resolve the fluctuative nature of thresholding. 
        context is an integer value determining the number
        of neighboring frames that should be used to decide
        if a frame is voiced.
        
        The log-energy values are subject to mean and var
        normalization to simplify the picking the right threshold. 
        In this framework, the default threshold is 0.0
        """
    xframes = zero_mean(xframes) #makes mean 0
    n_frames = xframes.shape[0]
    
    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames,1))
    for i in range(n_frames):
        start = max(i-context,0)
        end = min(i+context,n_frames-1)
        n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
        n_total = end-start+1
        xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr) #returns either 0 or 1
    return xvad

def grab_sound_detection_areas(wav,sample_rate, percent_high_nrg):
    win_len = int(sample_rate*0.025)
    hop_len = int(sample_rate*0.010)
    sframes = enframe(wav,win_len,hop_len)
    vad = nrg_vad(sframes,percent_high_nrg)
    vad_ar = deframe(vad,win_len,hop_len)
    return vad_ar


if __name__=='__main__':
    test_file='data/20120901004517.wav'
    fs,s = read_wav(test_file)
    s = s[0:fs]
    # plt.plot(s)
    # plt.show()
    # plt.close()
    win_len = int(fs*0.025) #window length = used for analysis rn 2.5% of sample
    hop_len = int(fs*0.010) # hop length = 
    sframes = enframe(s,win_len,hop_len) # rows: frame index, cols: each frame
    # plt.plot(sframes)#divides  into frames
    # plt.show()
    # plt.close()
    plot_this(compute_log_nrg(sframes))
    
    # percent_high_nrg is the VAD context ratio. It helps smooth the
    # output VAD decisions. Higher values are more strict.
    percent_high_nrg = 0.3 
    vad = nrg_vad(sframes,percent_high_nrg)
    #vad is 0 for no sound, 1 for sounds
    #now, vad is in frames 
    vad_str = deframe(vad,win_len,hop_len)
    plot_these(deframe(vad,win_len,hop_len),s)





