import pickle
from matplotlib import pyplot as plt
import numpy as np

filename = 'pca_1.pkl'
data = pickle.load(open(filename, 'rb'))

img = []
for compressor, data in data:
  layer = compressor.inverse_transform(data)
  img.append(layer / 255)
img = np.dstack(img)

plt.imshow(img)
plt.show()
  