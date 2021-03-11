#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function implements fuzz distortion on a given mono signal. 
Input parameters are the signal x and the amount of gain G. 

@author: nk
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
#%%

def Fuzz(x, G):
    # Normalize input:
    x = x / (np.max(np.abs(x)))
    # Apply pre-gain:
    x *= G
    # Oversampling:
    x_over = signal.resample(x, 8 * len(x))
    # Sign Vec:
    x_01 = np.sign(x_over)
    # Apply Fuzz:
    y_over = x_01 - x_01 * np.exp(-x_01 * x_over)
    # Downsampling:
    y = signal.decimate(y_over, 8)
    # Normalization:
    y /= np.max(np.abs(y))
    return y
#%%
if __name__ == "__main__":
    G = 5 # You can be generous with the gain.
       
    sr, data = wavfile.read("../TestGuitarPhraseMono.wav")
    y = Fuzz(data,G)
    
    wavfile.write("example_Fuzz.wav", sr, y.astype(np.float32))