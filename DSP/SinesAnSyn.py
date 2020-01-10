import numpy as np
import math
from scipy.fftpack import ifft, fftshift
from scipy.signal import blackmanharris, triang
import dftAnSyn as dft
import Utilities as U 

def sineModelAnal(x, fs, win, N, H, tresh, maxSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
	if (minSineDur <0):                          
		raise ValueError("Minimum duration is smaller than 0")
	
	hM1 = int(math.floor((win.size+1)/2))                     
	hM2 = int(math.floor(win.size/2))                         
	x = np.append(np.zeros(hM2),x)                          # center around 0
	x = np.append(x,np.zeros(hM2))                          
	begin = hM1                                                      
	end = x.size - hM1                                     
	win = win / sum(win)                                          
	tfreq = np.array([])
	while begin<end:                                                  
		x1 = x[begin-hM1:begin+hM2]                            
		mX, pX = dft.dftAnal(x1, win, N)                        
		peaksLocation = U.peakDetection(mX, tresh)                        
		iplocation, ipmag, ipphase = U.peak_parabolicInterp(mX, pX, peaksLocation)  
		ipfreq = fs*iplocation/float(N)                            
		# perform sinusoidal tracking by adding peaks to trajectories
		tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
		tfreq = np.resize(tfreq, min(maxSines, tfreq.size)) 
		tmag = np.resize(tmag, min(maxSines, tmag.size))    
		tphase = np.resize(tphase, min(maxSines, tphase.size)) 
		jtfreq = np.zeros(maxSines)                          
		jtmag = np.zeros(maxSines)                           
		jtphase = np.zeros(maxSines)                         
		jtfreq[:tfreq.size]=tfreq                             
		jtmag[:tmag.size]=tmag                                
		jtphase[:tphase.size]=tphase                         
		if begin == hM1:                                        # if first frame initialize output sine tracks
			xtfreq = jtfreq 
			xtmag = jtmag
			xtphase = jtphase
		else:                                                 # rest of frames append values to sine tracks
			xtfreq = np.vstack((xtfreq, jtfreq))
			xtmag = np.vstack((xtmag, jtmag))
			xtphase = np.vstack((xtphase, jtphase))
		begin += H
	xtfreq = cleanSineTracks(xtfreq, round(fs*minSineDur/H))  
	return xtfreq, xtmag, xtphase

def sineModelSynth(tfreq, tmag, tphase, N, H, fs):	
	hN = N//2                                              
	L = tfreq.shape[0]                                      
	out = 0                                            
	ysize = H*(L+3)                                     
	y = np.zeros(ysize)                                  
	synthWin = np.zeros(N)                                     
	triWin = triang(H*2)                                     
	synthWin[hN-H:hN+H] = triWin                                    
	bh = blackmanharris(N)               
	bh = bh / sum(bh)                                     
	synthWin[hN-H:hN+H] = synthWin[hN-H:hN+H]/bh[hN-H:hN+H]             
	lastytfreq = tfreq[0,:]                                 
	ytphase = 2*np.pi*np.random.rand(tfreq[0,:].size)     
	for l in range(L):                                    
		if (tphase.size > 0):                               
			ytphase = tphase[l,:] 
		else:
			ytphase += (np.pi*(lastytfreq+tfreq[l,:])/fs)*H     # propagate phases
		Y = U.genSinesSpectrum(tfreq[l,:], tmag[l,:], ytphase, N, fs)         
		lastytfreq = tfreq[l,:]                               # save frequency for phase propagation
		ytphase = ytphase % (2*np.pi)                        
		yw = np.real(fftshift(ifft(Y)))                      
		y[out:out+N] += synthWin*yw                              
		out += H                                     
	y = np.delete(y, range(hN))                         
	y = np.delete(y, range(y.size-hN, y.size))         
	return y
	
def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
	# tfreq: frequencies of tracks from previous frame

	tfreqn = np.zeros(tfreq.size)                             
	tmagn = np.zeros(tfreq.size)                       
	tphasen = np.zeros(tfreq.size)                           
	pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]   
	incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] 
	newTracks = np.zeros(tfreq.size, dtype=np.int) -1        
	magOrder = np.argsort(-pmag[pindexes])                    
	pfreqt = np.copy(pfreq)                                   
	pmagt = np.copy(pmag)                                      
	pphaset = np.copy(pphase)                              
	if incomingTracks.size > 0:                                
		for i in magOrder:                                       
			if incomingTracks.size == 0:                         
				break
			track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))  
			freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) 
			if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):   
					newTracks[incomingTracks[track]] = i                     
					incomingTracks = np.delete(incomingTracks, track)       
	indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   
	if indext.size > 0:
		indexp = newTracks[indext]                                   
		tfreqn[indext] = pfreqt[indexp]                       
		tmagn[indext] = pmagt[indexp]                     
		tphasen[indext] = pphaset[indexp]                   
		pfreqt= np.delete(pfreqt, indexp)                   
		pmagt= np.delete(pmagt, indexp)              
		pphaset= np.delete(pphaset, indexp)                        

	emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]      
	peaksleft = np.argsort(-pmagt)                               
	if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):  
			tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
			tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
			tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
	elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):  
			tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
			tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
			tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
			tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
			tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
			tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
	return tfreqn, tmagn, tphasen

def cleanSineTracks(tfreq, minTrackLength=3):
	if tfreq.shape[1] == 0:                              
		return tfreq
	nFrames = tfreq[:,0].size                            
	nTracks = tfreq[0,:].size                               # n of tracks in a frame
	for t in range(nTracks):                              
		trackFreqs = tfreq[:,t]                       
		trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0) & (trackFreqs[1:]>0))[0] + 1
		if trackFreqs[0]>0:
			trackBegs = np.insert(trackBegs, 0, 0)
		trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0) & (trackFreqs[1:] <=0))[0] + 1
		if trackFreqs[nFrames-1]>0:
			trackEnds = np.append(trackEnds, nFrames-1)
		trackLengths = 1 + trackEnds - trackBegs              
		for i,j in zip(trackBegs, trackLengths):              
			if j <= minTrackLength:
				trackFreqs[i:i+j] = 0
	return tfreq
