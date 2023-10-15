import numpy as np
import matplotlib.pyplot as plt


def f(x, a, b, c):
    return (a - 4) * x**2 + (b - 5) * x + (c - 6)


def g(x):
    return np.exp(x) / (np.exp(x) + 1)


index = [5, 5, 5]

x = np.linspace(-1.5, 1.5, 500)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Wykres funkcji")
ax.plot(x, f(x, index[0], index[1], index[2]), color="blue", lw=2)
ax.plot(x, g(x), color="red", lw=2)

plt.show()
