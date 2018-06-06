import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# x = np.array([1, 2, 3, 4, 5])
# y = np.array([7, 8, 2, 4, 1, 3, 4, 7, 8, 2, 1])

x = np.random.uniform(low=0, high=1, size=100)
y = np.random.uniform(low=0, high=1, size=100)

sq_x = x**2
sq_y = y**2

# print(z)
matrix = []

for i in sq_x:
    z = []
    for j in sq_y:
        sq = np.sqrt((abs(i - j)) / 2)
        z.append(sq)
    # print(z)
    matrix.append(z)
# print(matrix)

matrix_np = np.array(matrix)
print(matrix_np)

plt.set_cmap('gray')
plt.imshow(matrix_np, cmap='gray')