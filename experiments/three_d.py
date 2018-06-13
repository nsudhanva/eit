import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# your real data here - some 3d boolean array
img = plt.imread('eitcrop.png')
x, y, z = np.indices((10, 10, 10))
voxels = (x == y) | (y == z)

ax.voxels(voxels)

plt.show()