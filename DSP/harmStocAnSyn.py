import numpy as np
from scipy.signal import resample, blackmanharris, triang, hanning
from scipy.fftpack import fft, ifft, fftshift
import math
import harmonicAnSyn as HM
import SinesAnSyn as SM
import dftAnSyn as DFT
import stochasticAnSyn as STOC
import Utilities as U

def hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf):

	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
	xr = U.subtractSines(x, Ns, H, hfreq, hmag, hphase, fs)   	
	stocEnv = STOC.stochasticAnal(xr, H, H*2, stocf)
	return hfreq, hmag, hphase, stocEnv

def hpsModelSynth(hfreq, hmag, hphase, stocEnv, N, H, fs):

	yh = SM.sineModelSynth(hfreq, hmag, hphase, N, H, fs)
	yst = STOC.stochasticSynth(stocEnv, H, H*2)
	y = yh[:min(yh.size, yst.size)]+yst[:min(yh.size, yst.size)]
	return y, yh, yst


def hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, stocf):

	hM1 = int(math.floor((w.size+1)/2))
	hM2 = int(math.floor(w.size/2))
	Ns = 512
	H = Ns//4
	hNs = Ns//2      
	begin = max(hNs, hM1)
	end = x.size - max(hNs, hM1)
	fftbuffer = np.zeros(N)
	yhw = np.zeros(Ns)
	ystw = np.zeros(Ns)
	yh = np.zeros(x.size)
	yst = np.zeros(x.size)
	w = w / sum(w)
	sw = np.zeros(Ns)
	ow = triang(2*H)
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)
	bh = bh / sum(bh)
	wr = bh
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
	sws = H*hanning(Ns)/2
	hfreqp = []
	f0t = 0
	f0stable = 0
	while begin<end:  
	# analyze             
		x1 = x[begin-hM1:begin+hM2]
		mX, pX = DFT.dftAnal(x1, w, N)
		ploc = U.peakDetection(mX, t)        
		iploc, ipmag, ipphase = U.peak_parabolicInterp(mX, pX, ploc)
		ipfreq = fs * iploc/N
		f0t = U.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)
		if ((f0stable==0)&(f0t>0)) \
			or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
			f0stable = f0t
		else:
			f0stable = 0
		hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs)
		hfreqp = hfreq
		ri = begin-hNs-1 
		xw2 = x[ri:ri+Ns]*wr                   
		fftbuffer = np.zeros(Ns)    
		fftbuffer[:hNs] = xw2[hNs:]    
		fftbuffer[hNs:] = xw2[:hNs]                           
		X2 = fft(fftbuffer)
	# synth
		Yh = U.genSinesSpectrum(hfreq, hmag, hphase, Ns, fs)         
		Xr = X2-Yh
		mXr = 20 * np.log10(abs(Xr[:hNs]))
		mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf) # decimate the spectrum to avoid -Inf                     
		stocEnv = resample(mXrenv, hNs)
		pYst = 2*np.pi*np.random.rand(hNs)
		Yst = np.zeros(Ns, dtype = complex)
		Yst[:hNs] = 10**(stocEnv/20) * np.exp(1j*pYst)
		Yst[hNs+1:] = 10**(stocEnv[:0:-1]/20) * np.exp(-1j*pYst[:0:-1])

		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yh))
		yhw[:hNs-1] = fftbuffer[hNs+1:]
		yhw[hNs-1:] = fftbuffer[:hNs+1] 

		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yst))
		ystw[:hNs-1] = fftbuffer[hNs+1:]
		ystw[hNs-1:] = fftbuffer[:hNs+1]

		yh[ri:ri+Ns] += sw*yhw 
		yst[ri:ri+Ns] += sws*ystw 
		begin += H
	
	y = yh+yst
	return y, yh, yst