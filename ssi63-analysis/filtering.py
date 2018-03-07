import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.signal as sg

df = pd.read_hdf('ssi63.h5',start=20*60*60*10, stop=20*60*60*60)


h = df.altitude_barometer.values
from scipy.ndimage import gaussian_filter1d
a = gaussian_filter1d(np.diff(np.diff(sg.decimate(h,10))),500)
val = np.diff(df.valve_time_total.values[::20])
print([k for k in df.keys()])
plt.plot(val)
plt.show()
exit()


#plt.subplot(311)
plt.plot(df.altitude_barometer.values[::20])
plt.plot(df.altitude_gps.values[::20])
for i in np.where(val):
	plt.vlines(i,df.altitude_gps.min(),df.altitude_gps.max())
#plt.subplot(312)
#plt.plot(a)
#plt.subplot(313)
#plt.plot(val)
plt.show()