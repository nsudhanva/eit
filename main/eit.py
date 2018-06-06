import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

img = plt.imread(PARENT_DIR + '\\assets\\eitcrop.png')
img_two_d = img.mean(axis=2)

clrs = ('brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray')
cp = plt.contourf(img_two_d, colors=clrs)
plt.colorbar(cp)
plt.title('EIT Contour Plot')
plt.xlabel('Pixels(x)')
plt.ylabel('Pixels(y)')
plt.savefig(PARENT_DIR + '\\assets\\eit_contour_plot')
plt.show()