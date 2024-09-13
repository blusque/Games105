import numpy as np
import matplotlib.pyplot as plt

t = 1000
p = np.linspace(0, np.pi * 2, t)
beta = np.array([[3, 0], [0, -3], [-3, 0], [0, 3]])

w = np.mod(4 * p / (2 * np.pi), 1)
w = np.expand_dims(w, axis=-1)
k0 = np.mod(np.floor(4 * p / (2 * np.pi)) - 1, 4).astype(np.int32)
k1 = np.mod(np.floor(4 * p / (2 * np.pi)), 4).astype(np.int32)
k2 = np.mod(np.floor(4 * p / (2 * np.pi)) + 1, 4).astype(np.int32)
k3 = np.mod(np.floor(4 * p / (2 * np.pi)) + 2, 4).astype(np.int32)

alpha = np.zeros([t, 2])
alpha = beta[k1] + w * (0.5 * beta[k2] - 0.5 * beta[k0]) + \
    w ** 2 * (beta[k0] - 2.5 * beta[k1] + 
              2 * beta[k2] - 0.5 * beta[k3]) + \
    w ** 3 * (1.5 * beta[k1] - 1.5 * beta[k2] +
              0.5 * beta[k3] - 0.5 * beta[k0])

fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(alpha[:, 0], alpha[:, 1])
plt.show()
