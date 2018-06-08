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

df = pd.read_excel(PARENT_DIR + '\\assets\\datasets\\eit.xlsx', sheet_name="Sheet1", skiprows=[1, 2])

outlier = df['red'].quantile(0.99)