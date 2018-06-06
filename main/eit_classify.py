import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))

img = plt.imread(PARENT_DIR + '\\assets\\eitcrop.png')
img_two_d = img.mean(axis=2)
img_one_d = img_two_d.ravel()

colors = ['brown', 'red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray']
classify_dict = {}
low = 0.0
high = 1.1
skip = 0.1

intensity = np.arange(low, high, skip)
color_length = high / len(colors)
color_length_array = np.full((1, len(intensity) - 1), round(color_length, 2))
color_length_array = np.insert(color_length_array, 0, 0)
intensity_range = np.cumsum(color_length_array)

for index, color in enumerate(colors):
    # print(intensity_range[index], intensity_range[index + 1])
    classify_dict[color] = np.where(np.logical_and(img_one_d >= intensity_range[index], img_one_d < intensity_range[index + 1]))[0]

classify_dict_count = {}

for key, value in classify_dict.items():
    classify_dict_count[key] = len(value)

colors_tuple = tuple(classify_dict_count.keys())
y_pos = np.arange(len(colors_tuple))
pixels = classify_dict_count.values()

plt.bar(y_pos, pixels, align='center', alpha=0.5, color=colors_tuple)
plt.legend()
plt.xticks(y_pos, colors_tuple)
plt.ylabel('Pixels Count')
plt.title('Pixels vs Colors')

plt.show()