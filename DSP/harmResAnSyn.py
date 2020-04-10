import numpy as np
import math
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import harmonicAnSyn as HM
import dftAnSyn as DFT
import Utilities as U
import SinesAnSyn as S

def hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope):

	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
	Ns = 512
	xr = U.subtractSines(x, Ns, H, hfreq, hmag, hphase, fs)    
	return hfreq, hmag, hphase, xr
	
def hprModelSynth(hfreq, hmag, hphase, xr, N, H, fs):

	yh = S.sineModelSynth(hfreq, hmag, hphase, N, H, fs)        
	y = yh[:min(yh.size, xr.size)]+xr[:min(yh.size, xr.size)]   
	return y, yh
	
def hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et):

	hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
	Ns = 512                                                      # FFT size for synthesis (even)
	H = Ns//4                                                     # Hop size used for analysis and synthesis
	hNs = Ns//2      
	pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
	pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
	fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
	yhw = np.zeros(Ns)                                            # initialize output sound frame
	xrw = np.zeros(Ns)                                            # initialize output sound frame
	yh = np.zeros(x.size)                                         # initialize output array
	xr = np.zeros(x.size)                                         # initialize output array
	w = w / sum(w)                                                # normalize analysis window
	sw = np.zeros(Ns)     
	ow = triang(2*H)
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)
	bh = bh / sum(bh)
	wr = bh
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
	hfreqp = []
	f0t = 0
	f0stable = 0
	while pin<pend:  
	    # analyze             
		x1 = x[pin-hM1:pin+hM2]
		mX, pX = DFT.dftAnal(x1, w, N)
		ploc = U.peakDetection(mX, t)
		iploc, ipmag, ipphase = U.peak_parabolicInterp(mX, pX, ploc)
		ipfreq = fs * iploc/N
		f0t = U.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)
		if ((f0stable==0)&(f0t>0)) or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
			f0stable = f0t
		else:
			f0stable = 0
		hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs)
		hfreqp = hfreq
		ri = pin-hNs-1                                      
		xw2 = x[ri:ri+Ns]*wr
		fftbuffer = np.zeros(Ns)
		fftbuffer[:hNs] = xw2[hNs:]                                # zero-phase window 
		fftbuffer[hNs:] = xw2[:hNs]                     
		X2 = fft(fftbuffer)
		# synth
		Yh = U.genSinesSpectrum(hfreq, hmag, hphase, Ns, fs)
		Xr = X2-Yh                    
		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yh))
		yhw[:hNs-1] = fftbuffer[hNs+1:]
		yhw[hNs-1:] = fftbuffer[:hNs+1]
		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Xr))
		xrw[:hNs-1] = fftbuffer[hNs+1:]
		xrw[hNs-1:] = fftbuffer[:hNs+1]
		yh[ri:ri+Ns] += sw*yhw
		xr[ri:ri+Ns] += sw*xrw
		pin += H
	y = yh+xr
	return y, yh, xr
 
