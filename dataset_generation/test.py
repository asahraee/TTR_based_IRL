import numpy as np

p1 = np.array([[1, 2, 3],[4, 5, 6]])
p2 = np.array([[7, 8, 9, 13, 14],[10, 11, 12, 15, 16]])

p = np.concatenate((p1, p2), axis=1)

print(p)
