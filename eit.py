import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

img = plt.imread('eitcrop.png')
img_two_d = img.mean(axis=2)

c = ('brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray')
cp = plt.contourf(img_two_d, colors=c)
plt.colorbar(cp)
plt.title('Filled Contours Plot')
plt.xlabel('Pixels(x)')
plt.ylabel('Pixels(y)')
plt.show()