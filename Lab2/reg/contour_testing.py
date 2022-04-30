import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (X**2 + Y**2)

# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=True, fontsize=10)
# ax.set_title('Simplest default with labels')
# plt.show()

x = np.array([1.,2.,3.,4.])
print(x)
ones = np.ones((len(x), 1))
print(ones)
f = np.concatenate((ones, x), axis = 1)
print(f)
