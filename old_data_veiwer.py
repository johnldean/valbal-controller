import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
#import functions 
import pandas as pd 

df = pd.read_hdf('../ssi54.h5')
plt.plot(df.altitude_barometer)
for t in df.index.values[np.nonzero(np.diff(df.ballast_time_total))]:
	plt.axvline(t,c='r') 

for t in df.index.values[np.nonzero(np.diff(df.valve_time_total))]:
	plt.axvline(t, c='b')
plt.show()
plt.clf()
####
