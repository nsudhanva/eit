# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import matplotlib.pyplot as plt
import os

# Import to show plots in seperate Windows
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))

# Import image - converts image into a 3D numpy array
img = plt.imread(PARENT_DIR + '\\assets\\eit_images\\eitcrop.png')

# Convert the colored-3D image into grayscale-2D
img_two_d = img.mean(axis=2)

# Plot a contour plot based on the list of colors and save an image
clrs = ('brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray')
cp = plt.contourf(img_two_d, colors=clrs)
plt.colorbar(cp)
plt.title('EIT Contour Plot')
plt.xlabel('Pixels(x)')
plt.ylabel('Pixels(y)')
plt.savefig(PARENT_DIR + '\\assets\\eit_contour_plot')
plt.show()