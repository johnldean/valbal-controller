import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import functions as fn

		
p = [-0.5,.8]		#pole locations
z = [-0.5,.999]		#zero loations
gain = 2e-7			#gain 
freq = 0.001    	#at frequency 
Fs0 = 1 	    	#sample rate

a,b = fn.generic_biquad(p,z,gain,freq,Fs0)

print(a,b)
x = np.ones((1000));
y= fn.biquad_filter(x,a,b)
print(y[-1])
print(np.sum(b)/np.sum(a))