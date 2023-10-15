from mpl_toolkits.mplot3d import (
    Axes3D,
)
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)

Z = -(X**2 + Y**3)

surf = ax.plot_surface(  # type: ignore
    X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True  # type: ignore
)

plt.show()
