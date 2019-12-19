import numpy as np
import math
import dftAnSyn as dft

def stftAnal(x, win, N, H):

    if (H <= 0):
        raise ValueError("Hop size too small!")

    M = win.size
    pM1 = (M+1)//2
    pM2 = M//2
    x = np.append(np.zeros(pM2),x)
    x = np.append(x, np.zeros(pM2))
    begin = pM1        # init sound pointer in the middle of analysis window 
    end = x.size - pM1
    win = win / sum(win)
    xmX = []
    xphX = []
    while begin<=end:
        x1 = x[begin-pM1:begin+pM2]
        mX, phX = dft.dftAnal(x1,win,N)
        xmX.append(np.array(mX))
        xphX.append(np.array(phX))
        begin += H
    xmX = np.array(xmX)
    xphX = np.array(xphX)

    return xmX, xphX

def stftSynth(mX, phX, M, H):

    pM1 = (M+1)//2
    pM2 = M//2
    nFrames = mX[:,0].size
    y = np.zeros(nFrames*H + pM1 + pM2)
    begin = pM1
    for i in range(nFrames):
        y1 = dft.dftSynth(mX[i,:], phX[i,:], M)
        y[begin-pM1:begin+pM2] += H*y1
        begin += H
    
    # delete half of first window and end of sound added by stft analysis
    y = np.delete(y, range(pM2))
    y = np.delete(y, range(y.size-pM1, y.size))

    return y
