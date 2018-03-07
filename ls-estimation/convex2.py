import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import pandas as pd 
from matplotlib.colors import LogNorm
import cvxpy as cvx

def plex(x,y=None):
	if np.all(y) == None:
		plt.plot(x)
	else:
		plt.plot(x,y)
	plt.show()
	exit()	

df = pd.read_hdf('../ssi63-analysis/ssi63.h5',start=20*60*60*10, stop=20*60*60*120)

h = df.altitude_barometer.values
val = df.valve_time_total.values
bal = df.ballast_time_total.values

klin = 10
sample_time = 1 #sample time in minutes
intv = 1000
t = intv/20
navg = 20
rho = 1e-7

from scipy.ndimage import gaussian_filter1d
h = sg.decimate(sg.decimate(sg.decimate(h,10),10),10)
a = np.diff(np.diff(gaussian_filter1d(h,3)))

val = np.diff(val[::1000])/1000*0.001
bal = np.diff(bal[::1000])/1000*0.0006
val = val[1:]
bal = bal[1:]
val2 = np.convolve(val,np.ones(navg)/navg,mode='full')[:-(navg-1)] 
bal2 = np.convolve(bal,np.ones(navg)/navg,mode='full')[:-(navg-1)]

Vtf_st = -6
Vtf_end = 12
Btf_st = 0
Btf_end = 20



def method1():
	'''
	First, try and fine the TF for vent at a few good points, then used that to find the scaling 
	at other points. Works alright, not the best though.
	'''
	st = 500
	en = 1000
	T = en-st
	asm = a[st:en]
	valsm = val[st:en] 
	Vtf = cvx.Variable(Vtf_end-Vtf_st)
	Val = cvx.Variable(T)
	g=0
	for i in range(-Vtf_st,T-Vtf_end):
		if i+Vtf_st==0:
			g += valsm[i]*cvx.vstack(Vtf, np.zeros(T-Vtf_end+Vtf_st))
		else:
			g += valsm[i]*cvx.vstack(np.zeros((i+Vtf_st,1)),Vtf,np.zeros((T-Vtf_end-i,1)))
	const = [Val == g]
	obj = cvx.Minimize(cvx.sum_entries(cvx.huber(asm+Val,0.2)))
	prob = cvx.Problem(obj,const)
	prob.solve(solver = 'ECOS',verbose=True,reltol=1e-10,abstol=1e-10,max_iters=200)
	#plt.plot(Vtf.value)
	#plt.show()
	Vtf = Vtf.value

	T = a.size
	print(T,bal.size)
	Bal = cvx.Variable(T)
	Val = cvx.Variable(T)
	Btf = cvx.Variable(Btf_end-Btf_st)
	Vs = cvx.Variable(T)
	const = []
	f = 0
	for i in range(-Btf_st,T-Btf_end):
		if i+Btf_st==0:
			f += bal[i]*cvx.vstack(Btf, np.zeros(T-Btf_end+Btf_st))
		else:
			f += bal[i]*cvx.vstack(np.zeros((i+Btf_st,1)),Btf,np.zeros((T-Btf_end-i,1)))
	const.append(Bal == f)
	const.append(Btf >= 0)
	const.append(cvx.sum_entries(Btf) == 400)

	g=0
	for i in range(-Vtf_st,T-Vtf_end):
		if i+Vtf_st==0:
			g += Vs[i]*val[i]*cvx.vstack(Vtf, np.zeros(T-Vtf_end+Vtf_st))
		else:
			g += Vs[i]*val[i]*cvx.vstack(np.zeros((i+Vtf_st,1)),Vtf,np.zeros((T-Vtf_end-i,1)))
	const.append(Val == g)
	const.append(Vs >= 0)
	#const.append(cvx.sum_entries(Vtf) == 400)


	obj = cvx.Minimize(cvx.sum_entries(cvx.huber(a-Bal+Val,0.2)))
	prob = cvx.Problem(obj,const)
	print("OHP")
	r = prob.solve(solver = 'ECOS',verbose=True,reltol=1e-10,abstol=1e-10,max_iters=200)
	print(r)


	plt.subplot(411)
	plt.plot(bal)
	plt.plot(val)
	plt.subplot(412)
	plt.plot(a)
	plt.plot(a+np.array((-Bal.value+Val.value))[:,0])
	plt.subplot(413)
	plt.plot(Bal.value,'r--')
	plt.subplot(414)
	plt.plot(Btf.value)
	#plt.plot(V.value)
	plt.show()
	1