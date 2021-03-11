#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function applies distortion to a given audio signal x. It is based off the
physical modelling of a vaccuum tube distortion, but uses a stacked layout to 
achieve a result that is similar sounding to a "Muff" type distortion pedal, by
treating the positive and negative parts of the signal differently.

@author: nk
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile
#%%
def Muff(x,G=1,Qp=-0.1,Qn=-0.1,Dp=8,Dn=4,r1=0.97,r2=0.8):
    
    # Normalize input:
    x = x / (np.max(np.abs(x)))
    # Apply pre-gain:
    x *= G
    # Oversampling:
    x_over = signal.resample(x, 8 * len(x))
    # Positive part of signal:
    x_pos = 0.5 * (np.sign(x_over) * x_over + x_over)
    if Qp == 0:
        y_pos = x_pos/(1-np.exp(-Dp * x_pos)) - 1 / Dp
    else:
        # Apply Distortion:
        PLUS = Qp / (1-np.exp(Dp*Qp))
        EQUAL_QX = 1 / Dp + Qp / (1 - np.exp(Dp * Qp))
        # Logical indexing:
        logiQ = (x_pos % Qp != 0).astype(int)
        x_Q = x_pos - Qp
        y_0 = - (logiQ - 1) * EQUAL_QX
        y_1 = (logiQ * x_Q) / (1 - np.exp(-Dp * (logiQ * x_pos -Qp)))+PLUS
        y_pos = y_0 + y_1
    # Negative part of signal:
    x_neg = x_over - x_pos
    if Qn == 0:
        y_neg = x_neg / (1 - np.exp(-Dn * x_neg)) - 1 / Dn
    else:
        PLUS = Qn / (1-np.exp(Dn*Qn))
        EQUAL_QX = 1 / Dn + Qn / (1 - np.exp(Dn * Qn))
        logiQ = (x_neg % Qp != 0).astype(int)
        x_Q = x_neg - Qn
        y_0 = -(logiQ - 1) * EQUAL_QX
        y_1 = (logiQ * x_Q) / (1 - np.exp(-Dn * (logiQ * x_neg - Qn)))+PLUS
        y_neg = y_0 + y_1
    y_over = y_neg+y_pos
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
    Qp=-0.1
    Qn=-0.1
    Dp=8
    Dn=4
    r1=0.97
    r2=0.8
    
    sr, data = wavfile.read("../TestGuitarPhraseMono.wav")
    y = Muff(data,G,Qp,Qn,Dp,Dn,r1,r2)
    wavfile.write("example_Muff.wav", sr, y.astype(np.float32))