#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function applies distortion to a given audio signal by modelling the
effects from a vacuum tube. Input parameters are the signal vector x,
pre-gain G, "work point" Q, amount of distortion D, and pole positions r1
and r2 for the two filters used.

@author: nk
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
#%%
def Valve(x,G=1,Q=-0.05,D=400,r1=0.97,r2=0.8):
    
    # Normalize input:
    x = x / (np.max(np.abs(x)))
    # Apply pre-gain:
    x *= G
    # Oversampling:
    x_over = signal.resample(x, 8 * len(x))
    if Q == 0:
        y_over = x_over/(1-np.exp(-D * x_over)) - 1 / D
    else:
        # Apply Distortion:
        PLUS = Q / (1-np.exp(D*Q))
        EQUAL_QX = 1 / D + Q / (1 - np.exp(D * Q))
        # Logical indexing:
        logiQ = (x_over % Q != 0).astype(int)
        x_Q = x_over - Q
        y_0 = - (logiQ - 1) * EQUAL_QX
        y_1 = (logiQ * x_Q) / (1 - np.exp(-D * (logiQ * x_over -Q)))+PLUS
        y_over = y_0 + y_1
    # Downsampling:
    y = signal.decimate(y_over, 8)
    # Filtering:
    B = [1, -2, 1]
    A = [1, -2*r1, r1**2]
    y = signal.filtfilt(B, A, y)
    b = 1-r2
    a = [1, -r2]
    y = signal.filtfilt(b, a, y)
    # Normalization:
    y /= np.max(np.abs(y))
    return y
#%%
if __name__ == "__main__":
    G=1
    Q=-0.05
    D=400
    r1=0.97
    r2=0.8
    
    sr, data = wavfile.read("../TestGuitarPhraseMono.wav")
    y = Valve(data,G,Q,D,r1,r2)
    wavfile.write("example_Valve.wav", sr, y.astype(np.float32))