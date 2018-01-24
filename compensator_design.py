import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import functions as fn


def position():
	vmax = 2
	kl = 3.1
	lmax = (vmax/kl)**2
	lrange = np.arange(-lmax, lmax, 0.0001)[np.newaxis].T
	vrange = (np.sign(lrange)*kl*(np.abs(lrange))**0.5)
	k1 = np.linalg.pinv(lrange)@vrange
	plt.plot(lrange,vrange)
	plt.plot(lrange,k1*lrange)
	#print(k1)
	#plt.show()
	#plt.clf()

	Hs = ct.tf([0,0,k1],[1,0,0])
	#print(Hs)
	Hz = ct.matlab.c2d(Hs,1,method='zoh')
	#print(Hz)

	p = [-0.5,.8]		#pole locations
	z = [-0.5,.999]		#zero loations
	gain = 2e-7			#gain 
	freq = 0.001    	#at frequency 
	Fs0 = 1 	    	#sample rate
	a,b = fn.generic_biquad(p,z,gain,freq,Fs0)
	print('compensator',a,b)
	Kz = ct.tf(b,a,1)

	a,b = fn.biquad_lowpass(0.01,0.5,20)
	print('lowpass', a,b)

	a,b = fn.biquad_lowpass(0.005,0.707,1)
	Fz = ct.tf(b,a,1)

	ct.bode_plot(Hz*Kz*Fz,np.logspace(-4,-1,1000))
	ct.bode_plot(Hz*Kz,np.logspace(-4,-1,1000))
	plt.show()
	plt.clf()
	sys = ct.feedback(Hz*Kz*Fz,1)
	y, t = ct.step(sys,np.arange(0,10000,1))
	plt.plot(t,y.T)
	sys = ct.feedback(Hz*Kz,1)
	y, t = ct.step(sys,np.arange(0,10000,1))
	plt.plot(t,y.T)
	plt.show()


	'''
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
	'''

def speed(): 
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

	Hs = ct.tf([0,kl],[1,0])
	Hz = ct.matlab.c2d(Hs,1,method='zoh')

	p = [-.5,.99]		#pole locations
	z = [-.5,.999]	#zero loations
	gain = 0.002		#gain 
	freq = 0.1    	#at frequency 
	Fs = 1		#sample rate
	#fn.zplane(p,z)
	#plt.show()
	k = gain/np.abs( (1 - z[0]*np.exp(-freq/Fs*1j))*(1 - z[1]*np.exp(-freq/Fs*1j))/( (1 - p[0]*np.exp(-freq/Fs*1j))*(1 - p[1]*np.exp(-freq/Fs*1j))))
	b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
	a = [1, -(p[0]+p[1]), p[0]*p[1]]
	#print(a,b)
	Kz = ct.tf(b,a,1/Fs)

	ct.bode(Kz*Hz,np.logspace(-5,0))
	plt.show()

position()