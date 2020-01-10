import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
import math
import dftAnSyn as DFT
import SinesAnSyn as SM
import Utilities as U 

def f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et):

	if (minf0 < 0):
		raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")
	
	if (maxf0 >= fs/2):
		raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")
	
	if (H <= 0):
		raise ValueError("Hop size (H) smaller or equal to 0")

	hM1 = int(math.floor((w.size+1)/2))
	hM2 = int(math.floor(w.size/2))
	x = np.append(np.zeros(hM2),x)
	x = np.append(x,np.zeros(hM1))
	begin = hM1    
	end = x.size - hM1
	w = w / sum(w)
	f0 = []
	f0t = 0
	f0stable = 0 
	while begin<end:             
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
		f0 = np.append(f0, f0t)              
		begin += H                              
	return f0


def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):

	# Detect the harmonics of a frame from a set of spectral peaks

	if (f0<=0):            
		return np.zeros(nH), np.zeros(nH), np.zeros(nH)
	hfreq = np.zeros(nH)            
	hmag = np.zeros(nH)-100            
	hphase = np.zeros(nH)           
	hf = f0*np.arange(1, nH+1)
	hi = 0                         
	if hfreqp == []:                        
		hfreqp = hf
	while (f0>0) and (hi<nH) and (hf[hi]<fs/2):  
		pei = np.argmin(abs(pfreq - hf[hi]))             
		dev1 = abs(pfreq[pei] - hf[hi])               
		dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) 
		threshold = f0/3 + harmDevSlope * pfreq[pei]
		if ((dev1<threshold) or (dev2<threshold)):    
			hfreq[hi] = pfreq[pei]                       
			hmag[hi] = pmag[pei]                        
			hphase[hi] = pphase[pei]              
		hi += 1                            
	return hfreq, hmag, hphase

def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):

	if (minSineDur <0):  
		raise ValueError("Minimum duration of sine tracks smaller than 0")
		
	hM1 = int(math.floor((w.size+1)/2))                  
	hM2 = int(math.floor(w.size/2))
	x = np.append(np.zeros(hM2),x)
	x = np.append(x,np.zeros(hM2)) 
	begin = hM1         
	end = x.size - hM1 
	w = w / sum(w) 
	hfreqp = []   
	f0t = 0  
	f0stable = 0  
	while begin<=end:           
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
		hfreq, hmag, hphase = harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope) 
		hfreqp = hfreq
		if begin == hM1:  
			xhfreq = np.array([hfreq])
			xhmag = np.array([hmag])
			xhphase = np.array([hphase])
		else:                          
			xhfreq = np.vstack((xhfreq,np.array([hfreq])))
			xhmag = np.vstack((xhmag, np.array([hmag])))
			xhphase = np.vstack((xhphase, np.array([hphase])))
		begin += H        
	xhfreq = SM.cleanSineTracks(xhfreq, round(fs*minSineDur/H))  
	return xhfreq, xhmag, xhphase

def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et):

	hM1 = int(math.floor((w.size+1)/2)) 
	hM2 = int(math.floor(w.size/2))
	x = np.append(np.zeros(hM2),x)     
	x = np.append(x,np.zeros(hM1))   
	Ns = 512  
	H = int(Ns/4)                           
	hNs = int(Ns/2)      
	begin = max(hNs, hM1)                     
	end = x.size - max(hNs, hM1)                        
	fftbuffer = np.zeros(N)                     
	yh = np.zeros(Ns)                  
	y = np.zeros(x.size)      
	w = w / sum(w)                       
	sw = np.zeros(Ns)                   
	ow = triang(2*H)                           
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)            
	bh = bh / sum(bh)                                    
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]    
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
		hfreq, hmag, hphase = harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs) 
		hfreqp = hfreq
	# synth
		Yh = U.genSinesSpectrum(hfreq, hmag, hphase, Ns, fs)            
		fftbuffer = np.real(ifft(Yh))                   
		yh[:hNs-1] = fftbuffer[hNs+1:]                 
		yh[hNs-1:] = fftbuffer[:hNs+1] 
		y[begin-hNs:begin+hNs] += sw*yh              
		begin += H                                          
	y = np.delete(y, range(hM2))                    
	y = np.delete(y, range(y.size-hM1, y.size))         
	return y