import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct
import random as rd 
import pandas as pd

"""
Simulates a controller
"""

#df = pd.read_hdf('../ssi54.h5')

#data_h_ = df.altitude_barometer.values
#data_h_ = data_h_[500000:]
a = [1, 0.4, -0.05]
b = [0.0008955778634698831, -0.00043883315310024273, -0.00044331104241759211]
y = np.zeros(3)
x = np.zeros(3)

l = 0
v = 0
h = 14000
hT = 13500
bounds = 0
dl = 0
Fs = 20
kl = 5
t_ = np.arange(0,10000,1/Fs)
i = 0
dlb = 0.001
dlv = -0.001
Tminb = 2
Tminv = 2
tlastb = 0
tlastv = 0


#old controller
c = [0.6,0.001,0.00066,0.6,0.001,0.00066]
hlv = h
hlb = h
Dl_ = []
V_ = []
B_ = []
h_ = np.ones(1000)*h
v_ = []
dl_ = []
y_ = []
dh0_ = []
dlcmd_ = []


for t in t_:
	
	#linear compensator
	if t%1 == 0:
		y = np.roll(y,-1,0)
		x = np.roll(x,-1,0)
		x[2]= (hT - h) if np.abs(hT - h) > bounds else 0
		y[2] = (1/a[0]*(b[0]*x[2] + b[1]*x[2-1]  + b[2]*x[2-2] - a[1]*y[2-1] - a[2]*y[2-2]))
		dlcmd = y[2]*0.1
		dlcmd = dlcmd if np.abs(dlcmd) < 0.001 else np.sign(dlcmd)*0.001

		Twaitb = np.abs(dlb*Tminb / dlcmd) if dlcmd > 0 else np.infty 
		Twaitv = np.abs(dlv*Tminv / dlcmd) if dlcmd < 0 else np.infty

		if t-tlastb >= Twaitb:
			tlastb = t
		if t-tlastv >= Twaitv:
			tlastv = t
		#print(t-tlastv,t-tlastb, Twaitv,Twaitb)

	dl = 0
	if t-tlastb < Tminb:
		dl = dlb
	if t-tlastv < Tminv:
		dl = dlv
	#dl = dlcmd
	# old controller
	if t%15 == 0:
		p = np.polyfit(np.concatenate((np.arange(-1000/Fs,0,1/Fs),t_))[i:i+1000],h_[-1000:],1)
		dh0 = p[0]

		h0 = np.polyval(p,t)
		V = c[0]*dh0 + c[1]*(h0 - hT) + c[2]*(h0 - hlv)
		B = -c[3]*dh0 - c[4]*(h0 - hT) - c[5]*(h0 - hlb)
		Dl = 0
		if V > 1:
			hlv = h
			Dl = -0.001
		if B > 1: 
			hlv = h
			Dl = 0.001
		#print(V,B)
	Dl_.append(Dl)
	V_.append(V)
	B_.append(B)

	#l += Dl/Fs
	#l += rd.gauss(0,0.0003)
	l += dl/Fs 
	#v = kl*np.sign(l)*np.sqrt(np.abs(l)) + rd.gauss(0,0.3)
	v = 30*l
	#v += rd.gauss(0,0.3)
	h += v/Fs
	#h = data_h_[i]
	dl_.append(dl)
	dlcmd_.append(dlcmd)
	v_.append(v)
	h_ = np.append(h_,h)
	y_.append(y)
	dh0_.append(dh0)

	i += 1


h_ = h_[1000:]

plt.subplot(411)
plt.plot(t_,h_)
plt.ylabel('altitude')
plt.subplot(412)
plt.plot(t_,v_)
plt.ylabel('velocity')
plt.subplot(413)
plt.plot(t_,dl_)
plt.plot(t_, dlcmd_)
plt.ylabel('dl/dt')
plt.show()