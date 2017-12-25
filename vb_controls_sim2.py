import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct
import random as rd 
import pandas as pd
import multiprocessing as mlt
"""
Simulates a controller
"""

turb = np.load('ls-estimation/turb1.npy')
dlift = np.load('ls-estimation/lift1.npy')

def sim(N_trials,Plot=False):
	cost_ = [];
	rang_ = [];
	std_ = [];
	for i in range(N_trials):
		### Linear Controller Stuff ###
		## Compensator
		
		p = [-0.5,.8]		#pole locations
		z = [-0.5,.999]		#zero loations
		gain = 1e-7			#gain 
		freq = 0.001    	#at frequency 
		Fs0 = 1 	    	#sample rate
		k = gain/np.abs( (1 - z[0]*np.exp(-freq/Fs0 *1j))*(1 - z[1]*np.exp(-freq/Fs0*1j))/( (1 - p[0]*np.exp(-freq/Fs0*1j))*(1 - p[1]*np.exp(-freq/Fs0*1j))))
		b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
		a = [1, -(p[0]+p[1]), p[0]*p[1]]
		'''

		p = [-.5,.99]		#pole locations
		z = [-.5,.999]	#zero loations
		gain = 2e-6		#gain 
		freq = 0.1    	#at frequency 
		Fs = 1		#sample rate
		#fn.zplane(p,z)
		#plt.show()
		
		k = gain/np.abs( (1 - z[0]*np.exp(-freq/Fs*1j))*(1 - z[1]*np.exp(-freq/Fs*1j))/( (1 - p[0]*np.exp(-freq/Fs*1j))*(1 - p[1]*np.exp(-freq/Fs*1j))))
		b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
		a = [1, -(p[0]+p[1]), p[0]*p[1]]
		'''
		cmd_min = 0.00005
		cmd_max = 0.001

		## valbal phsical constants
		dlb = 0.001 		#dl/dt while balasting in kg/s
		dlv = -0.001		#dl/dt while venting in kg/s
		Tminb = 5		#minimum time to balast in s
		Tminv = 5			#minimum time to balast in s
		## Loop Variables
		y = np.zeros(3)		#biquad filter output history
		x = np.zeros(3)		#biquad filter input history
		tlastb = 0			#time since last balast event
		tlastv = 0			#time since last balast event


		### 'Gobal' variables ###
		## Presets
		hT = 13500			#target altitude			
		Fs = 20				#frequency running new data at
		kl = 5				#lift constant (determines velocity)
		klin = 5     		#linearized lift contant 
		kfb = 0.05/1000
		LIN_V = True
		## State
		v = 0				#velocity
		h = 13500		#altitude
		l = -h*kfb			#lift 
		dl = 0				#dl/dt rate
		## other
		i = 0				#loop iterator


		### legacy Controller Stuff ###
		c = [.8,0.001,0.00066,.8,0.001,0.00066]
		hlv = h
		hlb = h
		dl_legacy = 0 	#dl/dt from legacy controller

		### Variable Arrays ###
		t_ = np.arange(0,40000,1/Fs)
		h_ = np.concatenate((np.ones(1000)*h,np.zeros(t_.size)))
		v_ = np.zeros(t_.size)
		dl_ = np.zeros(t_.size)
		y_ = np.zeros(t_.size)
		dh0_ = np.zeros(t_.size)
		dlcmd_ = np.zeros(t_.size)


		h_last = h

		for t in t_:
			
			### Linear compensator ###
			if t%1 == 0:

				if abs(hT - h) > 250:
					gain_factor = 1+((abs(hT - h)-250)/50)
				else:
					gain_factor = 1
				gain_factor = 1
				y = np.roll(y,-1,0)
				x = np.roll(x,-1,0)
				x[2] =  (hT - np.average(h_[i:i+1000]))
				#x[2] =  gain + (h_[i+999-Fs] - h_[i+999])
				y[2] = (1/a[0]*(b[0]*x[2] + b[1]*x[2-1]  + b[2]*x[2-2] - a[1]*y[2-1] - a[2]*y[2-2]))
				#print(x[2],y[2],h_)
				dlcmd = y[2] * gain_factor
				dlcmd = dlcmd if np.abs(dlcmd) < cmd_max else np.sign(dlcmd)*cmd_max
				dlcmd = dlcmd if np.abs(dlcmd) > cmd_min else 0
				Twaitb = np.abs(dlb*Tminb / dlcmd) if dlcmd > 0 else np.infty 
				Twaitv = np.abs(dlv*Tminv / dlcmd) if dlcmd < 0 else np.infty
				
				## timers for vent/balast actions
				if t-tlastb >= Twaitb:
					tlastb = t
				if t-tlastv >= Twaitv:
					tlastv = t
				h_last = h
			
			## dl/dt setting
			dl = 0
			if t-tlastb < Tminb:
				dl = dlb
			if t-tlastv < Tminv:
				dl = dlv
			
			### Legacy controller ###
			if t%10 == 0:
				p = np.polyfit(np.concatenate((np.arange(-1000/Fs,0,1/Fs),t_))[i:i+1000],h_[i:i+1000],1)
				dh0 = p[0]

				h0 = np.polyval(p,t)
				V =  c[0]*dh0 + c[1]*(h0 - hT) + c[2]*(h0 - min(hlv,h+500) )
				B = -c[3]*dh0 - c[4]*(h0 - hT) - c[5]*(h0 - max(hlb,h-500) )
				dl_legacy = 0
				if V > 1:
					hlv = h
					dl_legacy = -0.0005
				if B > 1: 
					hlv = h
					dl_legacy = 0.0005
			#dl = dl_legacy			#toggle to turn on legacy controller

			### Simulated Valbal Flight ### 
			#l += get_lift(t)/Fs
			l += dl/Fs 
			ladj = l + (h*kfb)
			v = klin*(ladj) if LIN_V else kl*np.sign(ladj)*np.sqrt(np.abs(ladj))
			v += get_turb(t)/Fs
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
		
		cost = np.sum(np.abs(dl_))
		rang = np.max(h_) - np.min(h_)
		std = np.std(h_) 
		cost_.append(cost)
		rang_.append(rang)
		std_.append(std)
		
		if Plot:
			print(cost,rang)
			fig, ax2 = plt.subplots()
			ax2.plot(t_/60/60,dl_, 'r')
			ax2.plot(t_/60/60,dlcmd_, 'g')
			ax2.set_xlabel('time (hr)')
			ax2.set_ylabel('dl/dt')
			ax2.tick_params('y',colors='r')
			ax1 = ax2.twinx()
			ax1.plot(t_/60/60,h_, 'b-')
			ax1.set_ylabel('altitude')
			ax1.tick_params('y',colors='b')
			fig.tight_layout()
			plt.show()

	return np.mean(cost_) , np.mean(rang_), np.mean(std_)

def sim_wrapper(jank):
	N_trials = 1
	return sim(N_trials)

def get_lift(t):
	global dlift
	#rd.gauss(0,0.0002)
	return dlift[int(t/60)]/60

def get_turb(t):
	global turb
	#rd.gauss(0,0.8)
	return turb[int(t/60)]



if __name__ == '__main__':
	sim(1,Plot=True)


	'''
	with mlt.Pool(6) as pool:
		result = pool.map(sim_wrapper,range(6))
		print([np.sum(i)/len(i) for i in zip(*result)])
	'''

