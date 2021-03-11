#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function receives an mono input signal x and implements distortion
via a static non-linearity algorithm as per Doidic et al. (1998). Arguments are
the gain G, and the mode (symmetrical or antisymmetrical) to select the 
corresponding distortion algorithm.

@author: nk
"""
import numpy as np
from scipy.io import wavfile
#%%
def Doidic(x,G=1, mode="sym"):
    
    # Normalize input:
    x = x / (np.max(np.abs(x)))
    # Apply pre-gain:
    x *= G
    
    if mode == "sym":
        # Apply distortion:
        y = np.abs(2 * x) * x**2 * np.sign(x)
    
    elif mode == "anti":
        # Apply distortion:
        x = x / (np.max(np.abs(x)))
        x_1 = x
        x_2 = x
        x_3 = x
        
        for n in range(len(x)):
            if x[n] < -0.08905:
                x_1[n] = x[n]
            elif x[n] < 0.320018:
                x_2[n] = x[n]
            else:
                x_3[n] = x[n]
    
        x_1b = np.abs(x_1)-0.032847
        
        y_1 = -0.75 * (1-(1-x_1b)+x_1b/3)+0.01
        y_2 = -6.153 * x_2**2 + 3.9375 * x_2
        y_3 = 0.630035
        
        y = y_1 + y_2 + y_3
    
    # Normalization:
    y /= np.max(np.abs(y))
    return y

#%%
if __name__ == "__main__":
    G=2
    
    sr, data = wavfile.read("../TestGuitarPhraseMono.wav")
    y = Doidic(data,G,mode="anti")
    wavfile.write("example_DoidicAnti.wav", sr, y.astype(np.float32))