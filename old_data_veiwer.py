import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import pandas as pd 
import functions as fn

#df = pd.read_hdf('../ssi54.h5')
'''
h_ = np.zeros(len(df))
j = 0
for i in df.index:
	p = (df.raw_pressure_1[i] + df.raw_pressure_2[i] + df.raw_pressure_3[i] + df.raw_pressure_4[i])/4
	h = 44330*(1-(p/101325)**(1/5.255))
	h_[j] = h
	if j%10000 == 0:
		print(j/len(df))
	j += 1


'''
'''
np.save('ssi_54_avg',df.altitude_barometer.values)
np.save('ssi_54_v',df.valve_time_total.values)
np.save('ssi_54_b',df.ballast_time_total)
'''
x1 = np.load('ssi_54_avg.npy')
x1 = x1[5*20*60*60:-30*20*60*60]
t_ = np.arange(0,len(x1))/60/60/20
a,b = fn.biquad_lowpass_derivative(0.001,.5,20)
print(a,b)


dy = fn.biquad_ext_filter(x1,a,b)
fig, ax2 = plt.subplots()
ax2.plot(t_,x1,'g-')
ax1 = ax2.twinx()
ax1.plot(t_[20*60*60:],dy[20*60*60:])
ax1.plot(t_,df.ascent_rate.values[5*20*60*60:-30*20*60*60])
plt.show()


'''
plt.plot(df.altitude_barometer)
for t in df.index.values[np.nonzero(np.diff(df.ballast_time_total))]:
	plt.axvline(t,c='r') 
for t in df.index.values[np.nonzero(np.diff(df.valve_time_total))]:
	plt.axvline(t, c='b')
plt.show()
plt.clf()
####
'''