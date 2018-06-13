# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))

for i in range(1, 1001):
    # Generate data:
    x, y, z = 10 * np.random.random((3, 50))
    
    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
    
    clrs = ('brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray')
    plt.contourf(zi, colors=clrs)
    plt.axis('off')
    plt.savefig(PARENT_DIR + '\\assets\\eit_images\\' + "eit_" + str(i) + ".png", bbox_inches='tight')
         