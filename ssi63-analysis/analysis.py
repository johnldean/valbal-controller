import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd


#df = pd.read_hdf('ssi63.h5',start=20*60*60*100, stop=20*60*60*120)
#df = pd.read_hdf('ssi63.h5')
#np.save('ssi63_bal',df.ballast_time_total.values)
#np.save('ssi63_val',df.valve_time_total.values)
#print([k for k in df.keys()])
#exit()
def plot_flight():
	bal = np.diff(np.load('ssi63_bal.npy'))
	val = np.diff(np.load('ssi63_val.npy'))
	tlong = np.arange(val.size)/20/60/60
	temp = np.load('ssi63_temp.npy')[::20]
	h_cmd = np.load('ssi63_hcmd.npy')[::20]
	h = np.load('ssi63_h.npy')[::20]
	t_ = np.arange(h.size)/60/60
	fig = plt.figure(figsize=(10,2.5))
	from matplotlib import gridspec
	plt.subplots_adjust(hspace=0)
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
	ax0 = plt.subplot(gs[0])

	for i in np.nonzero(bal)[0]:
		ax0.axvline(tlong[i], c='b',alpha=0.02)
	for i in np.nonzero(val)[0]:
		ax0.axvline(tlong[i], c='g',alpha=0.02)

	ax0.plot(t_,h,'black',label='altitude')
	ax0.plot(t_,h_cmd,'--',color='black',label='cmd')
	ax0.set_ylim((11000,17000))
	ax0.set_xlim((7,121))
	ax0.set_ylabel('altitude (m)')
	ax0.grid()
	ax1 = plt.subplot(gs[1])
	ax1.plot(t_,temp,'black')
	ax1.set_xlim((7,121))
	ax1.set_ylabel('temp ($^{\circ}$c)')
	ax1.set_xlabel('flight time (h)')
	ax1.grid()
	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.savefig('flight.png')

def plot_osc():
	st = 100
	en = 120
	bal = np.diff(np.load('ssi63_bal.npy'))[20*60*60*st:20*60*60*en]
	val = np.diff(np.load('ssi63_val.npy'))[20*60*60*st:20*60*60*en]
	t = np.arange(val.size)/20/60/60
	temp = np.load('ssi63_temp.npy')[20*60*60*st:20*60*60*en]
	h_cmd = np.load('ssi63_hcmd.npy')[20*60*60*st:20*60*60*en]
	h = np.load('ssi63_h.npy')[20*60*60*st:20*60*60*en]
	v = nd.gaussian_filter1d(np.diff(h[::20]),300)
	ts = np.arange(v.size)/60/60
	
	fig, ax1 = plt.subplots(figsize=(10,5))
	for i in np.nonzero(bal)[0]:
		ax1.axvline(t[i], c='b',alpha=0.1)
	for i in np.nonzero(val)[0]:
		ax1.axvline(t[i], c='g',alpha=0.015)
	ax1.plot(t,h,'black')
	ax1.set_ylabel('altitude (m)')
	ax1.set_xlabel('flight time (hr)')
	#ax1.grid(color='black',alpha=0.3)
	ax2 = ax1.twinx()
	ax2.plot(ts,v,color='red') 
	ax2.yaxis.label.set_color('xkcd:blood red')
	ax2.tick_params(axis='y', colors='xkcd:blood red')
	ax2.spines['right'].set_color('xkcd:blood red')
	ax2.set_ylabel('velocity (m/s)')
	ax2.grid(color='xkcd:blood red',alpha=0.3)
	plt.tight_layout()
	plt.savefig('osc.png')



plot_osc()	
'''
num = 4
plt.subplot(num,1,1)
df.altitude_barometer[::20].plot()
plt.subplot(num,1,2)
df.spag_vent_time_interval[::20].plot()
df.spag_ballast_time_interval[::20].plot()
plt.semilogy()
plt.subplot(num,1,3)
df.spag_valve_interval_counter[::20].plot()
df.spag_ballast_interval_counter[::20].plot()
plt.subplot(num,1,4)
df.spag_v_dldt[::20].plot()
df.spag_b_dldt[::20].plot()
plt.show()
plt.subplot(211)
df.spag_vent_time_interval =df.spag_vent_time_interval + 0.1 
df.spag_vent_time_interval[::20].plot()
#df.spag_ballast_time_interval[::20].plot()
plt.semilogy()
plt.subplot(212)
df.raw_pressure_1[::20].plot()
plt.show()
'''