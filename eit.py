import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

img = plt.imread('eitcrop.png')
img_two_d = img.mean(axis=2)

#plt.hist(x, edgecolor='black', bins = list(np.arange(0.0, 1.0, 0.05)))
#plt.savefig('histogram.jpg')
#plt.show()

#plt.set_cmap('gray')
#plt.contour(grayed)
#plt.imshow(grayed, cmap='gray')
#plt.axis('off')
plt.clabel(cp, inline=True, fontsize=10)
plt.contour(img_two_d)
plt.show()