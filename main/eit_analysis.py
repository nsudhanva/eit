# Copyright (c) 2018, Faststream Technologies
# Author: Sudhanva Narayana

import numpy as np
import pandas as pd
import os

# Import to show plots in seperate Windows
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt5')

# CURR and PARENT directory constants
CURR_DIR = os.path.dirname(os.path.abspath('__file__'))
PARENT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))

# Import dataset ignoring headers
df = pd.read_csv(PARENT_DIR + '\\assets\\datasets\\eit.csv', index_col=[0], header = [0], skiprows= [1] ,skipinitialspace=True)
df_ranges = pd.read_csv(PARENT_DIR + '\\assets\\datasets\\eit.csv', index_col=[0], header = [0], skiprows= [0], skipinitialspace=True, nrows=0)
df_columns_ranges = list(df_ranges.columns)
df_columns_colors = list(df.columns)
df_means = df.mean()

target_series = []

# Create target_series list of booleans
for i, color in enumerate(df_columns_colors):
    target_series.append(df[color] > df_means[i])
    
target = np.array(target_series)
target = np.transpose(target[-4:])

target_bools = []

# Create target_bools which creates the final Series of target column
for i in range(len(target)):
    if np.sum(target[i]) >= 1:
        target_bools.append(1)
    else:
        target_bools.append(0)
        
target_bools = pd.Series(target_bools)

columns_tuple_list = []

# Tuple for creating columns for DataFrame
for color, intensity_range in zip(df_columns_colors, df_columns_ranges):
    columns_tuple_list.append((color, intensity_range))
    
# Final DataFrame to csv
df.columns = pd.MultiIndex.from_tuples(columns_tuple_list)
df['target'] = target_bools
df.to_csv(PARENT_DIR + '\\assets\\datasets\\' + 'eit_data.csv')