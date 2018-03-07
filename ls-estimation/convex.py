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

df = pd.read_hdf('../ssi63-analysis/ssi63.h5',start=20*60*60*20, stop=20*60*60*40)

h = df.altitude_barometer.values
val = df.valve_time_total.values
bal = df.ballast_time_total.values

klin = 10
sample_time = 1 #sample time in minutes
intv = int(20*60*sample_time)
t = intv/20
navg = 20
rho = 1e-7

h = h[::intv]

val = np.diff(val[::intv])/1000*0.001
bal = np.diff(bal[::intv])/1000*0.0006
val = np.append([0],val)
bal = np.append([0],bal)
val2 = np.convolve(val,np.ones(navg)/navg,mode='full')[:-(navg-1)] 
bal2 = np.convolve(bal,np.ones(navg)/navg,mode='full')[:-(navg-1)]
h = h - h[0]; 
T = h.size
print(T)

h = h/abs(np.mean(h))
dl = cvx.Variable(T)
l = cvx.Variable(T)
v = cvx.Variable(T)
Val = cvx.Variable(T)
Bal = cvx.Variable(T)
a = cvx.Variable(1)
b = cvx.Variable(1)
turb = cvx.Variable(T)
const = 	[l[1:] == l[:-1] + dl[:-1] + Bal[:-1] + Val[:-1],
			v[1:] == v[:-1] + 1/klin/4*60*(l[:-1] - v[:-1]) + turb[:-1],
			h[1:] == h[:-1] + v[:-1]]
const.append(Bal == a*bal2)
const.append(Val == a*val2)

obj = cvx.Minimize(0.001*cvx.sum_squares(turb) + 100*cvx.sum_squares(cvx.diff(dl)))
prob = cvx.Problem(obj,const)
r = prob.solve(solver = 'ECOS',verbose=True,reltol=1e-10,abstol=1e-10,max_iters=200)
print(r)

'''
plt.subplot(211)
plt.hist(turb.value)
plt.subplot(212)
plt.hist(np.diff(l.value,axis=0))
plt.show()
'''
plt.subplot(411)
plt.plot(h)
plt.ylabel('alt')
plt.subplot(412)
plt.plot(dl.value)
plt.ylabel('estimated net lift')
plt.subplot(413)
plt.plot(turb.value)
plt.ylabel('estimated turbulence')
plt.subplot(414)
plt.plot(bal,'b')
plt.plot(bal2,'b')

plt.plot(val,'r')
plt.plot(val2,'r')
try:
	plt.plot(Bal.value,'b--')
	plt.plot(Val.value,'b--')
except:
	pass
plt.show()


