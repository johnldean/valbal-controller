import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sg 
import control as ct 
import functions as fn

b = [0, 1.124988749873439e-9, 1.124977500042189e-9]
a = [1, -1.999969998200029, 0.999970000449996]

y = fn.biquad_filter(10*np.ones(6000*20),a,b)

plt.plot(np.arange(y.size)/20,y)
plt.show()



max{(TEMP_THRESH - TEMP) * TEMP_GAIN,0}

max{(VCAP_NOMINAL - VCAP_ACTUAL) * VCAP_GAIN,0}

max{(RB_COMM_INTERVAL - TIME_SINCE_LAST_COMM) * COMM_GAIN,0}


