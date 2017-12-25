import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import pandas as pd 


h = np.load('ssi_54_avg.npy')
v = np.load('ssi_54_v.npy')
b = np.load('ssi_54_b.npy')
kl = 5

st = 20*60*60*24
en = 20*60*60*24*2


for ds in (1,5):
	navg = 2;
	rho = 1e-9
	ds = 20*60*ds
	t = ds/20
	h1 = h[st:en:ds]

	vent = np.diff(v[st-1:en:ds])/1000*0.00001
	bal = np.diff(b[st-1:en:ds])/1000*0.000001

	vent = np.convolve(vent,np.ones(navg)/navg,mode='full')[:-(navg-1)] 
	bal = np.convolve(bal,np.ones(navg)/navg,mode='full')[:-(navg-1)]


	h1 = h1 - h1[0]; 

	T = h1.size
	print(T)
	A = np.zeros((T,T))
	for i in range(T):
		A += np.diag(np.ones(T-i)*(1/2 + i*2),-i)
	A = A*kl*t

	t_ = np.arange(0,T*t,t)
	plt.plot(t_,A@bal)
plt.show()