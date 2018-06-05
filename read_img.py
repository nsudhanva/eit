import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = plt.imread('eitcrop.png')
shape = img.shape[0]
numbers = list(range(1, shape))
plt.axis('off')
plt.imshow(img)

collapsed = img.mean(axis=2)
x = list(collapsed[0])
y = list(collapsed[1])
plt.hist2d(x, y)
plt.colorbar()
plt.show()

print(collapsed.shape)
plt.set_cmap('gray')
plt.imshow(collapsed, cmap='gray')
plt.axis('off')
plt.show()