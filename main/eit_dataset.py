# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import to show plots in seperate Windows
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
colors = ['brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray']

image_files_list = ['eit_1.jpg', 'eit_2.jpg', 'eit_3.jpg', 'eit_4.jpg']

# Initial setup of intensity range
low = 0.0
high = 1.1
skip = 0.1

for image_file in image_files_list:
    # Import image - converts image into a 3D numpy array
    img = plt.imread(PARENT_DIR + '\\assets\\eit_images\\' + image_file)
    
    # Convert the colored-3D image into grayscale-2D
    img_two_d = img.mean(axis=2)

    # Flatten 2D array to 1D array
    img_one_d = img_two_d.ravel()

    classify_dict = {}

    # Generate intensity range
    intensity = np.arange(low, high, skip)
    color_length = high / len(colors)
    color_length_array = np.full((1, len(intensity) - 1), round(color_length, 2))
    color_length_array = np.insert(color_length_array, 0, 0)

    # Calculate cumulative sum of average range of intensity
    intensity_range = np.cumsum(color_length_array)
    intensity_range_strings = []

    # Create a classify dict of colors mapping to their datapoints (array)
    for index, color in enumerate(colors):
        # print(intensity_range[index], intensity_range[index + 1])
        intensity_range_strings.append(str(round(intensity_range[index], 2)) + ' - ' + str(round(intensity_range[index + 1], 2)))
        classify_dict[color] = np.where(np.logical_and(img_one_d >= intensity_range[index], img_one_d < intensity_range[index + 1]))[0]

    # Create a count of classified dict
    classify_dict_count = {}

    for key, value in classify_dict.items():
        classify_dict_count[key] = [len(value)]

    print(classify_dict_count)
