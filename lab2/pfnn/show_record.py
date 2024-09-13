import matplotlib.pyplot as plt
import numpy as np
import os

filename = "./record_Sun_Dec_24_20_17_24_2023/lAnkle.npy"
with open(filename, 'rb') as f:
    fsz = os.fstat(f.fileno()).st_size
    out = np.load(f)
    while f.tell() < fsz:
        out = np.vstack((out, np.load(f)))

print(out.shape)
plt.plot(out[..., 0])
plt.show()
