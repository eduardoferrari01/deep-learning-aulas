import numpy as np
import matplotlib.pyplot as plt

a = -1
b = 4
c = 0.4


def plotline(a, b, c):
    x = np.linspace(-2, 4, 50)
    y = (-a * x - c) / b

    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.plot(x, y)
    plt.grid(True)


p1 = (2, 0.4)
p2 = (1, 0.6)
p3 = (3, -0.4)

plotline(a, b, c)

plt.plot(p1[0], p1[1], color='b', marker='o')

plt.plot(p2[0], p2[1], color='r', marker='o')

plt.plot(p3[0], p3[1], color='g', marker='o')

ret1 = a * p1[0] + b * p1[1] + c
ret2 = a * p2[0] + b * p2[1] + c
ret3 = a * p3[0] + b * p3[1] + c

print("ret1 = %.2f " % ret1)
print("ret2 = %.2f " % ret2)
print("ret3 = %.2f " % ret3)

plt.show()
