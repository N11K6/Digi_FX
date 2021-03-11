#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function receives an mono input signal x and applies the soft clipping
algorithm by Araya & Suyama (1996) to apply distortion via a gain factor G.

@author: nk
"""
import numpy as np
from scipy.io import wavfile
#%%
def ArayaSuyama(x,G=1):
    
    # Normalize input:
    x = x / (np.max(np.abs(x)))
    # Apply pre-gain:
    x *= G
    # Apply distortion:
    y = np.zeros(len(x))
    for n in range(len(x)):
        if np.abs(x[n]) <1:
            y[n] = 1.5 * x[n] * (1 - x[n]**2 / 3)
        else:
            y[n] = np.sign(x[n])
    # Normalization:
    y /= np.max(np.abs(y))
    return y

#%%
if __name__ == "__main__":
    G= 3
    
    sr, data = wavfile.read("../TestGuitarPhraseMono.wav")
    y = ArayaSuyama(data,G)
    wavfile.write("example_ArayaSuyama.wav", sr, y.astype(np.float32))