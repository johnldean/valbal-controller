import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct
import random as rd 

"""
Simulates a controller
"""

a = [1, 0.4, -0.05]
b = [0.0008955778634698831, -0.00043883315310024273, -0.00044331104241759211]
y = np.zeros(3)
x = np.zeros(3)

l = 0
v = 0
alt = 14500
cmd = 14000
dl = 0
Fs = 20
kl = 5
t_ = np.arange(0,10000,1/Fs)
alt_ = []
v_ = []
dl_ = []
y_ = []
for t in t_:
	if t%1 == 0:
		y = np.roll(y,-1,0)
		x = np.roll(x,-1,0)
		x[2]= cmd - alt
		y[2] = (1/a[0]*(b[0]*x[2] + b[1]*x[2-1]  + b[2]*x[2-2] - a[1]*y[2-1] - a[2]*y[2-2]))
		effort = y[2]
		effort = effort if np.abs(effort) < 0.002 else np.sign(effort)*0.002
		effort = effort if np.abs(effort) >= 0.002 else 0
		errort = round(effort,4)
		y[2] = effort
	dl = effort + rd.gauss(0,0.001)
	l += dl/Fs
	v = kl*np.sign(l)*np.sqrt(np.abs(l)) + rd.gauss(0,1)
	alt += v/Fs
	dl_.append(effort)
	v_.append(v)
	alt_.append(alt)
	y_.append(y[2])


plt.subplot(311)
plt.plot(t_/60/60,alt_)
plt.ylabel('altitude')
plt.subplot(312)
plt.plot(t_/60/60,v_)
plt.ylabel('velocity')
plt.subplot(313)
plt.plot(t_/60/60,dl_)
plt.plot(t_/60/60,y_)
plt.ylabel('dl/dt')
#plt.tight_layout()
plt.show()