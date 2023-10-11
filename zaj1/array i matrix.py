import numpy as np

A = np.array([[1, 2, 2], [4, 5, 6], [7, 8, 5]], dtype=float)
print("A:", A)
print(A**-1, "\n ")

B = np.matrix(A)

print("B:", B)
print(B**-1)
