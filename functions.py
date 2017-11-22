import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 


def zplane(p,z):
	"""
	Plot Z plane poles and zeros
	"""
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	plt.axis('equal')
	plt.axhline(color='grey')
	plt.axvline(color='grey')
	circ = plt.Circle((0,0), radius=1, color='b',fill=False) 
	plt.plot(np.real(z),np.imag(z),'ro',ms=10,mfc=(1,1,1,1),mew=2)
	plt.plot(np.real(p),np.imag(p),'rx', ms=10,mew=2)
	plt.xlabel('real')
	plt.ylabel('imag')
	ax.add_patch(circ)
	plt.plot()




##### Biquad stuff #####
def biquad_filter(x,a,b):
	"""
	Filter input x with differnce equation
	"""
	x = np.concatenate((x,np.zeros(3)))
	y = np.zeros(x.size)
	for i in range(0,x.size):
		y[i] = 1/a[0]*(b[0]*x[i] + b[1]*x[i-1]  + b[2]*x[i-2] - a[1]*y[i-1] - a[2]*y[i-2])
	return y[:-3]

def biquad_lowpass(w0,q,Fs):
	"""
	Calculates biquad coeffs for a 2nd order lowpass
	"""
	w0 = 2 * np.pi * f0 / Fs
	alpha = np.sin(w0)/(2*Q)

	b = [(1-np.cos(w0))/2, 1-np.cos(w0),(1-np.cos(w0))/2]
	a = [1+alpha, -2*np.cos(w0), 1-alpha]
	return(a,b)

def biquad_lowpass_derivative(f0,Q,Fs):
	"""
	Calculates difference eq coeffs for 2nd order biquad filter with differentiation
	Difference Equation usage:
	dy[i] = 1/a[0]*(b[0]*x[i] + b[1]*x[i-1]  + b[2]*x[i-2] + b[3]*x[i-3] - a[1]*dy[i-1] - a[2]*dy[i-2])
	
	Parameters
	----------
	f0
		Corner frequency of lowpass filter in Hz
	Q
		Quality factor 
	Fs
		Sample rate of filter
	
	Returns
	-------
	a
		array of 3 coefficents of feedback terms

	b
		array of 4 coefficents of forward terms

	"""

	#calculate biquad coeffs for 2nd order lowpass
	w0 = 2 * np.pi * f0 / Fs
	alpha = np.sin(w0)/(2*Q)
	b = [(1-np.cos(w0))/2, 1-np.cos(w0),(1-np.cos(w0))/2]
	a = [1+alpha, -2*np.cos(w0), 1-alpha]
	
	#take derivative
	b = np.subtract(b + [0] ,[0] + b) * Fs
	return(a,b)
