import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

kl = 3.2
klin = 10
b_dldt = 0.00057
dry_mass = 2.9
df = pd.read_hdf('ssi63_osc.h5')

b_mass = 5.5 - df.ballast_time_total.values[0]/1000*b_dldt


h_ = df.altitude_barometer.values
v_ = np.diff(df.valve_time_total.values)
b_ = np.diff(df.ballast_time_total.values)
bl_ = np.pad(np.convolve(b_/5000,np.ones(20*5)*b_dldt,mode='valid'),(20*5,0),mode='constant')
bf_ = np.convolve(b_,sg.gaussian(20*60*4,20*60/2),mode='same')
i_ = np.arange(0,h_.size)
t_ = i_/20/60
he_ = np.arange(0,h_.size)/h_.size*(h_[-1]-h_[0])
h2_ = h_ - he_
h_fft = np.fft.rfft(h2_)
h_fft = h_fft * sg.gaussian(2*h_fft.size-1,50)[h_fft.size-1:]
hf_ = np.fft.irfft(h_fft) + he_
v_ = np.diff(hf_)*20


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

for i in np.nonzero(b_)[0]:
	ax1.axvline(i, c='b')
ax1.plot(i_,h_,'red')
ax2.plot(v_,'green')
plt.show()


'''
num = 0
for loc1 in [slice(245000,345000),slice(496000,596000),slice(790000,890000),slice(1095000,1195000)]:
	v0_ = v_[loc1]
	h0_ = hf_[loc1]
	b0_ = bl_[loc1]
	t_ = np.arange(0,v0_.size)/20/60

	l1 = -(v0_[0]/kl)**2 
	v1 = v0_[0]
	v1_ = np.zeros(v0_.size)
	h1 = h0_[0]
	h1_ = np.zeros(v0_.size)
	for i in range(v0_.size):
		l1 += float(b0_[i])/20
		v1 += (-np.sign(v1)*(v1/kl)**2 + l1)/(b_mass+dry_mass)/20
		h1 += v1/20
		h1_[i] = h1
		v1_[i] = v1


	l2 = v0_[0]/klin 
	v2 = v0_[0]
	v2_ = np.zeros(v0_.size)
	h2 = h0_[0]
	h2_ = np.zeros(v0_.size)
	for i in range(v0_.size):
		l2 += float(b0_[i])/20
		v2 += (-v2/klin + l2)/(b_mass+dry_mass)/20
		h2 += v2/20
		h2_[i] = h2
		v2_[i] = v2

	plt.subplot(2,4,num+1)
	plt.plot(t_,h0_,)
	plt.plot(t_,h1_,'--')
	plt.plot(t_,h2_,'--')
	plt.xlabel('time (min)')
	plt.ylabel('altitude (m)')
	plt.title('Bottom of oscilation '+ str(num+1))
	if num == 0:
		plt.legend(['measured','quadratic drag','linear drag'])
	for i in np.nonzero(np.diff(b0_)>0)[0]:
		plt.axvline(t_[i], c='b')
	plt.subplot(2,4,num+5)
	plt.plot(t_,v0_,)
	plt.plot(t_,v1_,'--')
	plt.plot(t_,v2_,'--')
	plt.xlabel('time (min)')
	plt.ylabel('ascent rate (m/s)')
	num += 1

plt.show()

print(df.ballast_time_total[300*60*20]-df.ballast_time_total[0])

#l = -(v_[207*60]/kl)**2
l = v_[207*60]/klin
ve_ = []
ve = v_[207*60]
for t in t_[207*60:]:
	l += float(b[int(t*60-1)])/1000 * b_dldt
	#ve += (-np.sign(ve)*(ve/kl)**2 + l )/4
	ve += (-ve/klin + l)/2.5
	ve_.append(ve)

ax2.plot(t_[207*60:],ve_,'grey')
plt.show()
'''
