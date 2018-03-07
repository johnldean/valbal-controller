import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import pandas as pd 
import sys
sys.path.append('../')
import functions as fn


klin = 7
b_dldt = 0.00057

df = pd.read_hdf('ssi63.h5',start=20*60*60*100, stop=20*60*60*120)

b_ = np.diff(df.ballast_time_total.values)/1000*b_dldt*klin
valve_ = np.diff(df.valve_time_total.values)
h_ = df.altitude_barometer.values
t_ = np.arange(0,len(h_))/60/60/20
a,b = fn.biquad_lowpass_derivative(0.001,.5,20)
v_ = fn.biquad_ext_filter(h_,a,b)
p = [0,0.99994]
z = [-1,0]
k = 0.5
b = [k, -k*(z[0]+z[1]), k*z[0]*z[1]]
a = [1, -(p[0]+p[1]), p[0]*p[1]]
print(a,b)
e_ = fn.biquad_filter(b_,a,b)



he_ = np.arange(0,h_.size)/h_.size*(h_[-1]-h_[0])
h2_ = h_ - he_
h_fft = np.fft.rfft(h2_)
h_fft = h_fft * sg.gaussian(2*h_fft.size-1,50)[h_fft.size-1:]
hf_ = np.fft.irfft(h_fft) + he_
vg_ = np.diff(hf_)*20

vf_ = v_[:-1] + e_



fig, ax1 = plt.subplots()
ax1.axvline(t_[np.nonzero(b_)[0][0]], c='b',alpha=0.2,label='balast')
for i in np.nonzero(b_)[0][1:]:
	ax1.axvline(t_[i], c='b',alpha=0.2)
ax1.axvline(t_[np.nonzero(valve_)[0][0]], c='g',label='alt')
for i in np.nonzero(valve_)[0][1:]:
	ax1.axvline(t_[i], c='g')
ax1.plot(t_,h_,'b',label='alt')
ax1.set_xlabel('time (hr)')
ax1.set_ylabel('altitude (m)')
ax2 = ax1.twinx()
ax2.set_ylabel('velocity (m/s)')
ax2.plot(t_[20*60*60:-1],vf_[20*60*60:],'orange',label='fused')
ax2.plot(t_[:-1],vg_,'red',label='noncausal')
ax2.plot(t_[20*60*60:],v_[20*60*60:],'purple',label='causal nonfused')
ax2.plot(t_,df.ascent_rate.values,'pink',label='regression')
#ax2.axhline(0.4, c='orange',alpha=0.5)
ax2.legend()
fig.tight_layout()
plt.grid()
plt.show()


'''
plt.plot(df.altitude_barometer)
for t in df.index.values[np.nonzero(np.diff(df.ballast_time_total))]:
	plt.axvline(t,c='r')2
for t in df.index.values[np.nonzero(np.diff(df.valve_time_total))]:
	plt.axvline(t, c='b')
plt.show()
plt.clf()
####
'''