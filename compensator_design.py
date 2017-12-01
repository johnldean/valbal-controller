import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import functions as fn

vmax = 1
kl = 5
lmax = (vmax/kl)**2
lrange = np.arange(-lmax, lmax, 0.0001)[np.newaxis].T
vrange = (np.sign(lrange)*kl*(np.abs(lrange))**0.5)
k1 = np.linalg.pinv(lrange)@vrange
#plt.plot(lrange,vrange)
#plt.plot(lrange,k*lrange)
#plt.show()
#plt.clf()

Hs = ct.tf([0,0,k1],[1,0,0])
#print(Hs)
Hz = ct.matlab.c2d(Hs,1,method='zoh')
print(Hz)





## Design for theoretical optimum effort minimization
kk = 0.01
p = [-.5,.5]		#pole locations
z = [-.5,1-kk]	#zero loations
gain = 1		#gain 
freq = 0.1 + 0.1j     	#at frequency 
Fs = 1		#sample rate
b = [1, -(z[0]+z[1]), z[0]*z[1]]
a = [1, -(p[0]+p[1]), p[0]*p[1]]
Kz = ct.tf(b,a,1/Fs)
k =  1/np.abs(ct.evalfr(Kz*Hz, 1-kk + kk*1j))
b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
a = [1, -(p[0]+p[1]), p[0]*p[1]]
Kz = ct.tf(b,a,1/Fs)
ct.root_locus(Kz*Hz,np.arange(0,1,0.01))
plt.clf()

sys = ct.feedback(Kz*Hz,1)
y, t = ct.step(sys,np.arange(0,1000,1))
plt.plot(t,y.T)
plt.show()
