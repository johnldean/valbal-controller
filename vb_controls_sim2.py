import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct
import random as rd 
import pandas as pd

"""
Simulates a controller
"""

### Linear Controller Stuff ###
## Compensator
p = [-0.5,.5]		#pole locations
z = [-0.5,.999]		#zero loations
gain = 1.5e-7		#gain 
freq = 0.001     	#at frequency 
Fs0 = 1	 	    	#sample rate
k = gain/np.abs( (1 - z[0]*np.exp(-freq/Fs0 *1j))*(1 - z[1]*np.exp(-freq/Fs0*1j))/( (1 - p[0]*np.exp(-freq/Fs0*1j))*(1 - p[1]*np.exp(-freq/Fs0*1j))))
b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
a = [1, -(p[0]+p[1]), p[0]*p[1]]
## valbal phsical constants
dlb = 0.001 		#dl/dt while balasting in kg/s
dlv = -0.001		#dl/dt while venting in kg/s
Tminb = 2			#minimum time to balast in s
Tminv = 2			#minimum time to balast in s
## Loop Variables
y = np.zeros(3)		#biquad filter output history
x = np.zeros(3)		#biquad filter input history
tlastb = 0			#time since last balast event
tlastv = 0			#time since last balast event


### 'Gobal' variables ###
## State
l = 0				#lift 
v = 0				#velocity
h = 14000			#altitude
dl = 0				#dl/dt rate
## Presets
hT = 13500			#target altitude			
Fs = 20				#frequency running new data at
kl = 5				#lift constant (determines velocity)
klin = 31			#linearized lift contant 
## other
i = 0				#loop iterator



### legacy Controller Stuff ###
c = [0.6,0.001,0.00066,0.6,0.001,0.00066]
hlv = h
hlb = h
dl_legacy = 0 	#dl/dt from legacy controller

### Variable Arrays ###
t_ = np.arange(0,20000,1/Fs)
h_ = np.concatenate((np.ones(1000)*h,np.zeros(t_.size)))
v_ = np.zeros(t_.size)
dl_ = np.zeros(t_.size)
y_ = np.zeros(t_.size)
dh0_ = np.zeros(t_.size)
dlcmd_ = np.zeros(t_.size)


for t in t_:
	
	### Linear compensator ###
	if t%1 == 0:
		y = np.roll(y,-1,0)
		x = np.roll(x,-1,0)
		x[2]= (hT - h) 
		y[2] = (1/a[0]*(b[0]*x[2] + b[1]*x[2-1]  + b[2]*x[2-2] - a[1]*y[2-1] - a[2]*y[2-2]))
		dlcmd = y[2]*np.abs(x[2]/500)**2
		dlcmd = dlcmd if np.abs(dlcmd) < 0.001 else np.sign(dlcmd)*0.001
		dlcmd = dlcmd if np.abs(dlcmd) > 0.0001 else 0
		Twaitb = np.abs(dlb*Tminb / dlcmd) if dlcmd > 0 else np.infty 
		Twaitv = np.abs(dlv*Tminv / dlcmd) if dlcmd < 0 else np.infty

		## timers for vent/balast actions
		if t-tlastb >= Twaitb:
			tlastb = t
		if t-tlastv >= Twaitv:
			tlastv = t
	
	## dl/dt setting
	dl = 0
	if t-tlastb < Tminb:
		dl = dlb
	if t-tlastv < Tminv:
		dl = dlv
	
	### Legacy controller ###
	if t%15 == 0:
		p = np.polyfit(np.concatenate((np.arange(-1000/Fs,0,1/Fs),t_))[i:i+1000],h_[i:i+1000],1)
		dh0 = p[0]

		h0 = np.polyval(p,t)
		V =  c[0]*dh0 + c[1]*(h0 - hT) + c[2]*(h0 - hlv)
		B = -c[3]*dh0 - c[4]*(h0 - hT) - c[5]*(h0 - hlb)
		dl_legacy = 0
		if V > 1:
			hlv = h
			dl_legacy = -0.001
		if B > 1: 
			hlv = h
			dl_legacy = 0.001
	#dl = dl_legacy			#toggle to turn on legacy controller

	### Simulated Valbal Flight ### 
	l += rd.gauss(0,0.0002)
	l += dl/Fs 
	#v = kl*np.sign(l)*np.sqrt(np.abs(l)) + rd.gauss(0,0.3)
	v = klin*l
	v += rd.gauss(0,0.7)
	h += v/Fs
	

	### Store Variables ###
	#h = data_h_[i]
	dl_[i] = dl
	dlcmd_[i] =  dlcmd
	v_[i] = v
	h_[i+1000] = h
	y_[i] = y[2]
	dh0_[i] = dh0

	i += 1


h_ = h_[1000:]

print(np.sum(np.abs(dl_)))

plt.subplot(311)
plt.plot(t_/60/60,h_)
plt.ylabel('altitude')
plt.subplot(312)
plt.plot(t_/60/60,v_)
plt.ylabel('velocity')
plt.subplot(313)
plt.plot(t_/60/60,dl_)
#plt.plot(t_/60/60, dlcmd_)
#plt.legend(['actual', 'command'])
#plt.ylabel('dl/dt')
#plt.xlabel('time (hrs)')
plt.tight_layout()
plt.show()
#plt.savefig('linear_controller_example.png')