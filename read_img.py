import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

matrix1 = np.random.uniform(low=0, high=1, size=100)
matrix2 = np.random.uniform(low=0, high=1, size=100)

matrix = np.array([matrix1, matrix2])

img = plt.imread('eitcrop.png')
collapsed = img.mean(axis=2)

x = collapsed.ravel()
y = img.ravel()


#plt.hist(x, edgecolor='black', bins = list(np.arange(0.0, 1.0, 0.05)))
#plt.savefig('histogram.jpg')
#plt.show()

plt.set_cmap('gray')
plt.imshow(collapsed, cmap='gray')
plt.axis('off')
plt.show()