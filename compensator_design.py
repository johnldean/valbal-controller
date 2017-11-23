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
print(Hs)
Hz = ct.matlab.c2d(Hs,1,method='zoh')
print(Hz)

p = [-0.5,.1]		#pole locations
z = [-0.5,1]		#zero loations
gain = 0.00001		#gain 
freq = 0.001     	#at frequency 
Fs = 1		#sample rate
fn.zplane(p,z)
plt.show()
k = gain/np.abs( (1 - z[0]*np.exp(-freq/Fs*1j))*(1 - z[1]*np.exp(-freq/Fs*1j))/( (1 - p[0]*np.exp(-freq/Fs*1j))*(1 - p[1]*np.exp(-freq/Fs*1j))))

b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
a = [1, -(p[0]+p[1]), p[0]*p[1]]
print(a,b)
Kz = ct.tf(b,a,1/Fs)


#ct.root_locus(Kz*Hz,np.arange(0,2,0.0001),Plot = True)
#fig = plt.figure(1)
#ax = fig.add_subplot(111)	
#plt.axis('equal')
#circ = plt.Circle((0,0), radius=1, color='r',fill=False) 
#plt.axhline(color='grey')
#plt.axvline(color='grey')
#ax.add_patch(circ)
#plt.show()
ct.bode_plot(Kz*Hz)
plt.show()

'''
sys = ct.feedback(Kz*Hz,1)
t1,y1 = ct.step_response(Hz,np.arange(0,10,1))
t2,y2 = ct.step_response(Hs,np.arange(0,10,0.001))

y3 = []
yy1 = 0
yy2 = 0
xx = 0
for t in np.arange(0,10,0.05):
	yy2 += yy1*0.05
	yy1 += xx*float(k1)*0.05
	y3.append(yy2)
	xx = 1
print(y3)
plt.plot(t1,y1)
plt.plot(t2,y2)
plt.plot(np.arange(0,10,0.05),y3)
plt.show()
'''
