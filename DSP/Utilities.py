import numpy as np 

def peakDetection(mX, tresh):
    treshold = np.where(np.greater(mX[1:-1],tresh), mX[1:-1],0)
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1],0)
    previous_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)
    peak_locations = treshold * next_minor * previous_minor
    peak_locations = peak_locations.nonzero()[0] + 1
    return peak_locations

def peak_parabolicInterp(mX, pX, peaks):
    mags = mX[peaks]
    lmag = mX[peaks-1]
    rmag = mX[peaks+1]
    c_peak = peaks + 0.5 *(lmag-rmag)/(lmag-2*mags+rmag)
    pMag = mags - 0.25 * (lmag-rmag) * (c_peak - peaks)
    pPhase = np.interp(c_peak, np.arange(0, pX.size), pX)
    return c_peak, pMag, pPhase

def genSinesSpectrum(freq, mag, phase, N, fs):
    Y = np.zeros(N, dtype = complex)                 
    hN = N//2                                       
    for i in range(0, freq.size):                  # generate all sine spectral lobes
        loc = N * freq[i] / fs                     # it should be in range [0,hN-1] excluded
        if loc==0 or loc>hN-1: continue
        binremainder = round(loc)-loc
        lb = np.arange(binremainder-4, binremainder+5) # main lobe bins to read
        lmag = genBhLobe(lb) * 10**(mag[i]/20)        
        b = np.arange(round(loc)-4, round(loc)+5, dtype='int')
        for m in range(0, 9):
            if b[m] < 0:                                 
                Y[-b[m]] += lmag[m]*np.exp(-1j*phase[i])
            elif b[m] > hN:                             
                Y[b[m]] += lmag[m]*np.exp(-1j*phase[i])
            elif b[m] == 0 or b[m] == hN:                
                Y[b[m]] += lmag[m]*np.exp(1j*phase[i]) + lmag[m]*np.exp(-1j*phase[i])
            else:                                        
                Y[b[m]] += lmag[m]*np.exp(1j*phase[i])
        Y[hN+1:] = Y[hN-1:0:-1].conjugate()            
    return Y

# blackman-harris window main lobe
def genBhLobe(x):
    N = 512                                                 
    f = x*np.pi*2/N                                         
    df = 2*np.pi/N
    y = np.zeros(x.size)                                    
    consts = [0.35875, 0.48829, 0.14128, 0.01168]           # window constants
    for m in range(0,4):                                    
        y += consts[m]/2 * (sinc(f-df*m, N) + sinc(f+df*m, N))  
    y = y/N/consts[0]                                       
    return y

# main lobe of a sinc function 
def sinc(x, N):
    y = np.sin(N * x/2) / np.sin(x/2)                  
    y[np.isnan(y)] = N                                 
    return y

def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):

	# ef0max: maximum error allowed
	# f0t: f0 of previous frame (if stable), for smoother tracking
    if (minf0 < 0):
	    raise ValueError("Min f0 freq smaller than 0")
    if (maxf0 >= 10000):
	    raise ValueError("Max f0 freq bigger than 10000Hz")
    if (pfreq.size < 3) & (f0t == 0):
        return 0
    
    f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0]
    if (f0c.size == 0):
        return 0
    f0cf = pfreq[f0c]
    f0cm = pmag[f0c]
    
    if f0t>0:
        shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc]%f0t
        if maxcfd > f0t/2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd>(f0t/4)):
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]
    
    if (f0cf.size == 0):
        return 0
        
    f0, f0error = twoWayMismatch(pfreq, pmag, f0cf)
    
    if (f0>0) and (f0error<ef0max):
        return f0
    else:
        return 0

def twoWayMismatch(pfreq, pmag, f0c):

	# Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)

	p = 0.5                                          # weight freq
	q = 1.4                                          # weight mag
	r = 0.5                                          # scale mags
	rho = 0.33                                       # weight MP error
	Amax = max(pmag)      
	maxnpeaks = 10                             
	harmonic = np.matrix(f0c)
	ErrorPM = np.zeros(harmonic.size)            
	MaxNPM = min(maxnpeaks, pfreq.size)
	for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
		difmatrixPM = harmonic.T * np.ones(pfreq.size)
		difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
		FreqDistance = np.amin(difmatrixPM, axis=1)   
		peakloc = np.argmin(difmatrixPM, axis=1)
		Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
		PeakMag = pmag[peakloc]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
		harmonic = harmonic+f0c

	ErrorMP = np.zeros(harmonic.size)                
	MaxNMP = min(maxnpeaks, pfreq.size)
	for i in range(0, f0c.size) :                    # measured to predicted mismatch error
		nharm = np.round(pfreq[:MaxNMP]/f0c[i])
		nharm = (nharm>=1)*nharm + (nharm<1)
		FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
		Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
		PeakMag = pmag[:MaxNMP]
		MagFactor = 10**((PeakMag-Amax)/20)
		ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

	Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
	f0index = np.argmin(Error)                      
	f0 = f0c[f0index]                                

	return f0, Error[f0index]