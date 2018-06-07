# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import matplotlib.pyplot as plt
import os

# Import to show plots in seperate Windows
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))

# Import image - converts image into a 3D numpy array
img = cv2.imread(PARENT_DIR + '\\assets\\eit_images\\eitcrop.png')



