
import numpy as np
import pandas as pd
import random
from random import shuffle

df = pd.DataFrame()

ages = np.random.randint(low=10, high=80, size=(1000,))
sexes = ['male', 'female'] * 500
weights = np.arange(2, 110)

# variation = 10 - 20

# male ages - 10 - 17, weight - 30 - 50
# male ages - 18 - 70, weight - 50 - 100
# female ages - 10 - 17, weight - 25 - 45 
# female ages - 18 - 70, weight - 50 - 90

# 1 : 10 - 35.
# 2 : 30 - 55
# 3 : 25 - 45
# 4 : 35 - 55
# 5 : 35 - 55
# 6 : 45 - 65
# 7 : 35 - 55
# 8 : 45 - 65

# 1-2 : 35- 55
# 2-3 : 25 - 45
# 3-4 : 30 - 50
# 4-5 : 35 - 55
# 5-6 : 20 - 40
# 6-7 : 30 - 50
# 7-8 : 20 - 40
# 8-1 : 30 - 50

random.shuffle(sexes)
random.shuffle(ages)

final_weights = []

for i, j in zip(ages, sexes):
    if j == 'male':
        if i >= 10 and i <= 17:
            final_weights.append(random.sample(list(weights[30:50]), 1)[0])
        else:
            final_weights.append(random.sample(list(weights[50:100]), 1)[0])
    else:
        if i >= 10 and i <= 17:
            final_weights.append(random.sample(list(weights[25:45]), 1)[0])
        else:
            final_weights.append(random.sample(list(weights[50:90]), 1)[0])