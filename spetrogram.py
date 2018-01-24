import scipy.signal as sg 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm 

data = np.load('data.npy')
x = data[300000:-300000,1]
x = np.concatenate((x[:int(x.size/2)],x[int(x.size/2):]))
t = data[300000:-300000,0]/60/60
t = t - min(t)
Fs = 20
(f,ts,Z) = sg.stft(x,Fs,nperseg=4096*8,detrend='linear',boundary='even')
Z = np.abs(Z[:,0:-1])
ts = ts[0:-1]/60/60
Z = Z[:,:]
f = f[:]
#plt.subplot(211)
#plt.pcolor(ts, f, np.abs(Z),norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
#plt.colorbar()
plt.xlabel('time (hr)')
plt.ylabel('frequency (Hz)')
plt.subplot(212)
#plt.plot(t,x)
plt.xlim(np.min(ts),np.max(ts))
plt.xlabel('time (hr)')
#plt.show()
plt.clf()



'''
plt.subplot(211)
plt.plot(ts,np.mean(Z[1:100,:],axis=0))
plt.xlim(np.min(ts),np.max(ts))
plt.xlabel('Time (hrs)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(t,x)
plt.xlim(np.min(ts),np.max(ts))
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel('Time (hrs)')
plt.ylabel('Altitude (m)')
plt.show()
#plt.clf()
'''
'''
amps = np.mean(Z[100:500,:],axis=0)
alts = [x[np.argmin(np.abs(t - i))] for i in ts]
plt.xlabel('Altitude (m)')
plt.ylabel('Amplitude')

plt.title('Amplitude of high freqency oscilations vs Altitude')
plt.plot(alts,amps,'r*')
plt.plot(alts,0.00*np.exp(np.array(alts)/8000))
plt.show()
'''