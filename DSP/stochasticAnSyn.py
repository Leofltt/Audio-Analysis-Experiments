import numpy as np
from scipy.signal import hanning, resample
from scipy.fftpack import fft, ifft
import Utilities as U

def stochasticAnal(x, H, N, sf):

	hN = N//2+1                                            
	No2 = N//2                                             
	if (hN*sf < 3):                                     
		raise ValueError("The stochastic decimation factor is too small")
		
	if (sf > 1):                                        
		raise ValueError("The stochastic decimation factor is above 1")
		
	if (H <= 0):                                           
		raise ValueError("The hop size is smaller or equal to 0")

	if not(U.isPow2(N)):                                
		raise ValueError("The FFT size is not a power of 2")
		
	w = hanning(N)                                          
	x = np.append(np.zeros(No2),x)                          
	x = np.append(x,np.zeros(No2))                          
	begin = No2                                                     
	end = x.size-No2                                       
	while begin<=end:                         
		xw = x[begin-No2:begin+No2] * w                           
		X = fft(xw)
		absX = abs(X[:hN])
		absX[absX<np.finfo(float).eps] = np.finfo(float).eps                                 
		mX = 20 * np.log10(absX)
		mY = resample(np.maximum(-200, mX), int(sf*hN))    
		if begin == No2:                                        
			stocEnv = np.array([mY])
		else:                                                 
			stocEnv = np.vstack((stocEnv, np.array([mY])))
		begin += H                                              
	return stocEnv

def stochasticSynth(stocEnv, H, N):

	if not(U.isPow2(N)):                                 	
		raise ValueError("The FFT size is not a power of two")
 
	hN = N//2+1                                                 
	No2 = N//2                                                  
	frames = stocEnv[:,0].size                                    	
	ysize = H*(frames+3)                                         	
	y = np.zeros(ysize)                                     	
	ws = 2*hanning(N)                                        	
	pout = 0                                                 	
	for l in range(frames):                    
		mY = resample(stocEnv[l,:], hN)                        
		pY = 2*np.pi*np.random.rand(hN)                        
		Y = np.zeros(N, dtype = complex)                       
		Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                   
		Y[hN:] = 10**(mY[-2:0:-1]/20) * np.exp(-1j*pY[-2:0:-1]) 
		fftbuffer = np.real(ifft(Y))                           
		y[pout:pout+N] += ws*fftbuffer                         
		pout += H  
	y = np.delete(y, range(No2))                              
	y = np.delete(y, range(y.size-No2, y.size))               
	return y

	
