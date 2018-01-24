import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_hdf('ssi63.h5',start=20*60*60*70, stop=20*60*60*120)

h = df.altitude_barometer.values
h  = h - np.arange(0,h.size)/h.size*(h[-1]-h[0])
H = np.fft.rfft(h)

H[:100] = 0

plt.subplot(211)
plt.plot(h)
plt.subplot(212)
plt.plot(np.fft.ifft(H))

plt.show()