import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import functions as fn

a,b = fn.generic_biquad([0,0.8],[0,0.99],1,1,1)

print(a,b)
x = np.ones((1000));
y= fn.biquad_filter(x,a,b)
print(y[-1])
print(np.sum(b)/np.sum(a))