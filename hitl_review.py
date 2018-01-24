import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_hdf('data93.bin_19min.h5')
print([val for val in df.keys()])
df.spag_effort442.plot()
plt.show()