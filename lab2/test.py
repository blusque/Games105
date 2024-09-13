import matplotlib.pyplot as plt
import numpy as np

v1 = 6
v2 = 1
l1 = 45
l2 = 100

ss = []

for i in range(2, 10000):
    # alpha = np.linspace(0, 1, i)
    alpha1 = np.linspace(0, 1, i)
    alpha2 = np.linspace(0, 1, i + 1)
    # alpha1 = ((2 * alpha1 - 1) ** 3 + 1) / 2
    # alpha2 = ((2 * alpha2 - 1) ** 3 + 1) / 2
    # s1 = ((1 - alpha) * v1 + alpha * v2) / \
    #     ((1 - alpha) * v1 * l1 + alpha * v2 * l2)
    # s2 = ((1 - alpha) * v1 + alpha * v2 + v1) / \
    #     ((1 - alpha) * v1 * l1 + alpha * v2 * l2 + v1 * l1)
    s1 = ((1 - alpha1) * v1 + alpha1 * v2) / \
        ((1 - alpha1) * v1 * l1 + alpha1 * v2 * l2)
    s2 = ((1 - alpha2) * v1 + alpha2 * v2) / \
        ((1 - alpha2) * v1 * l1 + alpha2 * v2 * l2)
    # print(s.shape)
    s = s2.sum() - s1.sum()
    # s = s1.sum()
    ss.append(s)

print(ss)
ss = np.array(ss)
x = np.linspace(2, 9999, 9998)
alpha = np.linspace(0, 1, 200)
alpha = ((2 * alpha - 1) ** 3 + 1) / 2
plt.plot(alpha)
plt.show()
plt.plot(x, ss)
plt.show()
