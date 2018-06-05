import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

writer = pd.ExcelWriter('eit.xlsx')
matrix = np.random.uniform(low=1, high=20, size=(100, 100))
numbers = list(range(1, 101))
attributes = ['R' + str(i) for i in numbers]
print(attributes)
df = pd.DataFrame(matrix)
df.columns = attributes
df.index = attributes
print(df.head())
df.to_excel(writer, 'Sheet1')

plt.imshow(df, interpolation='bicubic')
plt.colorbar()
plt.show()
