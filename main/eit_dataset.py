# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import pandas as pd
import cv2
import os

# Import to show plots in seperate Windows
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt5')

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
colors = ['brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray']
colors = colors[::-1]
colors_p = [i + '_%' for i in colors]

image_files_list = []

for i in range(1, 1001):
    image_files_list.append('eit_' + str(i) + '.png')

# Initial setup of intensity range
low = 0
high = 255
skip = 1
   
classify_dict = {}
classify_dict_per = {}

# Generate intensity range
intensity = np.arange(low, high, skip)
color_length = high / len(colors)
color_length_array = np.full((1, len(intensity) - 1), round(color_length, 2))
color_length_array = np.insert(color_length_array, 0, 0)

# Calculate cumulative sum of average range of intensity
intensity_range = np.cumsum(color_length_array)

for c, p in zip(colors, colors_p):
    classify_dict[c] = []
    classify_dict_per[p] = []
    
for image_file in image_files_list:
    # Import image - converts image into a 3D numpy array
    # img = cv2.imread(PARENT_DIR + '\\assets\\eit_images\\' + image_file)
    
    # Import the colored-3D image into grayscale-2D
    img_two_d = cv2.imread(PARENT_DIR + '\\assets\\eit_images\\' + image_file, 0)

    # Flatten 2D array to 1D array
    img_one_d = img_two_d.ravel()

    total_length = len(img_one_d)
    intensity_range_strings = []

    # Create a classify dict of colors mapping to their datapoints (array)
    for index, (color, color_p) in enumerate(zip(colors, colors_p)):
        # print(intensity_range[index], intensity_range[index + 1])
        intensity_range_strings.append(str(round(intensity_range[index], 2)) + ' - ' + str(round(intensity_range[index + 1], 2)))
        intensity_range_length = len(np.where(np.logical_and(img_one_d >= intensity_range[index], img_one_d < intensity_range[index + 1]))[0])
        percentage = (intensity_range_length/total_length) * 100
        percentage = round(percentage, 2)
        classify_dict[color].append(intensity_range_length)
        classify_dict_per[color_p].append(percentage)

columns_tuple_list = []
# print(classify_dict)
for color, intensity_range in zip(colors, intensity_range_strings):
    columns_tuple_list.append((color, intensity_range))
    # print(color, intensity_range)

# Created tuples for DataFrames 
columns_p_tuple = list(zip(*[iter(colors_p)]*1, ['100'] * 8))
columns_tuple_list.sort(key=lambda tup: tup[0])

# DataFrame for values
df = pd.DataFrame(classify_dict)
df.columns = pd.MultiIndex.from_tuples(columns_tuple_list)
df = df[colors]
df.to_csv(PARENT_DIR + '\\assets\\datasets\\' + 'eit.csv')

# DataFrame for percentages
df_p = pd.DataFrame(classify_dict_per)
df_p.columns = pd.MultiIndex.from_tuples(columns_tuple_list)
df_p = df_p[colors]
df_p.to_csv(PARENT_DIR + '\\assets\\datasets\\' + 'eit_p.csv')
