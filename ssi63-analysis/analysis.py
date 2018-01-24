import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

t1 = 24
t2 = 48
#df = pd.read_hdf('ssi63.h5',start=20*60*60*100, stop=20*60*60*120)
df = pd.read_hdf('ssi63.h5')
df.altitude_barometer[::20].plot()
plt.show()


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