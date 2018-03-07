import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from scipy.stats import norm
from scipy.stats import laplace
import sys
sys.path.append('../')
import functions as fn

t1 = 25
t2 = 35
df = pd.read_hdf('../ssi63-analysis/ssi63.h5',start=20*60*60*t1, stop=20*60*60*t2)

h = df.altitude_barometer.values

he = np.arange(0,h.size)/h.size*(h[-1]-h[0])
h2 = h - he
h_fft = np.fft.rfft(h2)
h_fft = h_fft * sg.gaussian(2*h_fft.size-1,90)[h_fft.size-1:]
hf = np.fft.irfft(h_fft) + he
v = np.diff(h)*20
vf = np.diff(hf)*20
e = v - vf
e = e[20000:-20000]
e = e[abs(e - np.mean(e)) < 5*np.std(e)]
l = np.diff(vf)
l = l[abs(l - np.mean(l)) < 2.5*np.std(l)]
u1,o1 = norm.fit(e)
u2,o2 = norm.fit(l)

n,bins,patches = plt.hist(e,bins=100,normed=True)
x = np.arange(-0.4,0.4,0.001)
x2 = np.concatenate((np.linspace(-1,-0.1,100),np.linspace(0.1,1,100)))
plt.plot(x2,1.6*laplace.pdf(x2,0,.05),)
plt.plot(x,norm.pdf(x,u1,o1*0.87))
plt.show()

exit()

R = 0.000312**2
Q = 7.43901262832e-08**2
x = 0
P = 1
vfg = np.zeros(v.size)
ps = np.zeros(v.size)
for i,vx in enumerate(v):
	x = x
	P = P + Q
	K = P/(P+R)
	x = x + K*(vx - x)
	P = (1-K)*P
	vfg[i] = x
	ps[i] = P

a,b = fn.biquad_lowpass_derivative(0.0019,.5,20)
vfb = fn.biquad_ext_filter(h,a,b)
vfb[0:50000] = 0



print(u1,o1,np.std(e))
print(u2,o2)
print(np.sum((vf[50000:]-vfg[50000:])**2)**0.5/vf.size,np.sum((vf[50000:]-vfb[50000:-1])**2)**0.5/vf.size)
plt.subplot(311)
plt.plot(v)
plt.plot(vf)
plt.plot(vfg)
plt.plot(vfb)
plt.legend(['raw v', 'noncausaul gaussian filt v','bayes v','biquad v'])
#plt.plot(ps)

plt.subplot(312)
n,bins,patches = plt.hist(e,bins=100,normed=True)
x = np.arange(-0.4,0.4,0.001)
plt.plot(x,1/2*np.exp(-np.abs(x)))
plt.plot(x,norm.pdf(x,u1,o1))
plt.subplot(313)
plt.hist(l,bins=500,normed=True)
x = np.arange(-0.000075,0.000075,0.000001)
plt.plot(x,norm.pdf(x,u2,o2))
plt.show()
print