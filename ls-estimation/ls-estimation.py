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
ds = 20*60*1
t = ds/20
navg = 10
rho = 1e-9

h = h[st:en:ds]

vent = np.diff(v[st-1:en:ds])/1000*0.00001
bal = np.diff(b[st-1:en:ds])/1000*0.00001

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
A = A*kl*t**2
B = B*t
C = np.hstack((np.identity(T - 1), np.zeros((T-1,1)))) - np.hstack( (np.zeros((T-1,1)), np.identity(T - 1))) 

G1 = np.hstack((A,B,np.zeros(A.shape)))
G2 = np.hstack((C.T@C,np.zeros(A.shape),A.T))
G3 = np.hstack((np.zeros(A.shape),rho*np.identity(T),B.T))

G = np.vstack((G1,G2,G3))
print(G)

b = np.concatenate((h1 - A@bal, np.zeros(2*T)))[np.newaxis].T

x = np.linalg.inv(G)@b

l = x[0:T]
v = x[T:2*T]

plt.plot(h1 - A@bal)
plt.plot(h1)
plt.show()

plt.subplot(411)
plt.plot(h)
plt.subplot(412)
plt.plot(l) 
plt.subplot(413)
plt.plot(v)
plt.subplot(414)
plt.plot(vent)
plt.plot(bal)
#plt.show()