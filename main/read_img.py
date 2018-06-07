# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import matplotlib.pyplot as plt
import os

# Import to show plots in seperate Windows
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# ROOT and PARENT directory constants
ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

# Generate two 1D vectors of Uniform float - range 0 to 1 of size 100
matrix1 = np.random.uniform(low=0, high=1, size=100)
matrix2 = np.random.uniform(low=0, high=1, size=100)

# Create a 2D matrix out of the two 1D vectors
matrix = np.array([matrix1, matrix2])

# Import image - converts image into a 3D numpy array
img = plt.imread(PARENT_DIR + '\\assets\\eitcrop.png')

# Convert the colored-3D image into grayscale-2D
grayscale = img.mean(axis=2)

# Flatten 2D array to 1D array
x = grayscale.ravel()
y = img.ravel()

# Plot a histogram on 'x' array
# plt.hist(x, edgecolor='black', bins = list(np.arange(0.0, 1.0, 0.05)))
# plt.savefig(PARENT_DIR + '\\assets\\eitcrop.png')
# plt.show()

# Generate an image based on the 'grayscale' 2D matrix
plt.set_cmap('gray')
plt.imshow(grayscale, cmap='gray')
plt.axis('off')
plt.show()