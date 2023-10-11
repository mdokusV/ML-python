import numpy as np

X = np.matrix([[1, 2, 3], [1, 3, 6]], dtype=float)
y = np.matrix([5, 6], dtype=float)
y = y.T

out = ((X.T * X) ** -1) * X.T * y
print(out)
