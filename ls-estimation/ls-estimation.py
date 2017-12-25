import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import pandas as pd 
from matplotlib.colors import LogNorm


h = np.load('ssi_54_avg.npy')
v = np.load('ssi_54_v.npy')
b = np.load('ssi_54_b.npy')
klin = 30
sample_time = 1 #sample time in minutes


st = 20*60*60*24
en = int(20*60*60*24*7/4)
intv = 20*60*sample_time
t = intv/20
navg = 20
rho = 1e-8

h = h[st:en:intv]

vent = np.diff(v[st-1:en:intv])/1000*0.001
bal = np.diff(b[st-1:en:intv])/1000*0.0006

vent = np.convolve(vent,np.ones(navg)/navg,mode='full')[:-(navg-1)] 
bal = np.convolve(bal,np.ones(navg)/navg,mode='full')[:-(navg-1)]


h1 = h - h[0]; 

T = h.size
print(T)
B = np.ones((T,T))
B = np.tril(B)
A = np.zeros((T,T))
for i in range(T):
	A += np.diag(np.ones(T-i)*(1/2 + i*2),-i)
A = A*klin*t
B = B*t
C = np.hstack((np.identity(T - 1), np.zeros((T-1,1)))) - np.hstack( (np.zeros((T-1,1)), np.identity(T - 1))) 

G1 = np.hstack((A,B,np.zeros(A.shape)))
G2 = np.hstack((C.T@C,np.zeros(A.shape),A.T ))
G3 = np.hstack((np.zeros(A.shape),rho*np.identity(T),B.T ))

G = np.vstack((G1,G2,G3))

b = np.concatenate((h1 - A@bal + A@vent, np.zeros(2*T)))[np.newaxis].T

x = np.linalg.inv(G)@b

l = x[0:T]
v = x[T:2*T]

#plt.plot(h1 - A@bal)
#plt.plot(h1)
#plt.show()


plt.subplot(411)
plt.plot(h)
plt.subplot(412)
plt.plot(l) 
plt.subplot(413)
plt.plot(v)
plt.subplot(414)
plt.plot(vent)
plt.plot(bal)
plt.show()


l_adj = l[:,0] - np.arange(0,l.size)/l.size*(l[-1]-l[0])
l_fft = np.fft.rfft(l_adj)
l_fft[0] = 0
v_fft = np.fft.rfft(v[:,0])
f_ = np.fft.rfftfreq(l_adj.size,sample_time*60)

'''
for i in range(9):
	rand = np.random.ranf(l_fft.size)*2*np.pi
	randp = (np.cos(rand) + np.sin(rand)*1j) 
	l_fft_gen = np.abs(l_fft)*randp
	l_gen = np.fft.irfft(l_fft_gen)
	plt.subplot(10,2,2*i+1)
	plt.plot(l_gen)

	rand = np.random.ranf(l_fft.size)*2*np.pi
	randp = (np.cos(rand) + np.sin(rand)*1j)
	v_fft_gen = np.abs(v_fft)*randp
	v_gen = np.fft.irfft(v_fft_gen)
	plt.subplot(10,2,2*(i+1))
	plt.plot(v_gen)

plt.subplot(10,2,19)
plt.plot(l)
plt.subplot(10,2,20)
plt.plot(v)
plt.show()


(f,ts,Z) = sg.stft(h,1/60,nperseg=100,noverlap=(90),detrend='linear',boundary='even')
print(Z.shape)
plt.pcolor(ts/60/60, f, np.abs(Z[20:,:]))
plt.xlabel('time (hr)')
plt.ylabel('frequency (Hz)')

plt.show()
'''