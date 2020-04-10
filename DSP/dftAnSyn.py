import numpy as np
import math
from scipy.fftpack import fft, ifft
import Utilities as U


def dftAnal(x, win, N):

    if not(U.isPow2(N)):
        raise ValueError("FFT size not a power of 2")
    if (N < win.size):                                       
	    raise ValueError("FFT size smaller than window size")

    pN = (N//2) + 1
    pM1 = (win.size + 1)//2
    pM2 = win.size//2
    fftBuffer = np.zeros(N)
    win = win / sum(win)
    xw = x * win
    fftBuffer[:pM1] = xw[pM2:]
    fftBuffer[-pM2:] = xw[:pM2]
    X = fft(fftBuffer)
    absX = abs(X[:pN])
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps
    X[:pN].real[np.abs(X[:pN].real) < 1e-14] = 0.0
    X[:pN].imag[np.abs(X[:pN].imag) < 1e-14] = 0.0

    phX = np.unwrap(np.angle(X[:pN]))
    dbmX = 20 * np.log10(absX)

    return dbmX, phX


def dftSynth(mX, phX, n):

    pN = mX.size
    N = (pN-1)*2

    if not(U.isPow2(N)):
        raise ValueError("FFT size not a power of 2")

    pM1 = int(math.floor((n+1)/2))
    pM2 = int(math.floor(n/2))
    fftBuffer = np.zeros(n)
    y = np.zeros(n)
    Y = np.zeros(N, dtype = complex)
    Y[:pN] = 10**(mX/20) * np.exp(1j * phX)
    Y[pN:] = 10**(mX[-2:0:-1]/20) * np.exp(-1j*phX[-2:0:-1])
    fftBuffer = np.real(ifft(Y))
    y[:pM2] = fftBuffer[-pM2:]
    y[pM2:] = fftBuffer[:pM1]

    return y
