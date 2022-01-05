from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib import pyplot as plt
import numpy as np
import sys, pickle

def compress_image(data, method, n_components=5):
  new_img = []
  compression = []

  for i in range(3):
    if method == 'pca':
      compressor = PCA(n_components=n_components)
    elif method == 'svd':
      compressor = TruncatedSVD(n_components=n_components)
    data = img[:, :, i]
    compressor.fit(data)
    
    compressed = compressor.fit_transform(data) # save this? - pickle, numpy
    reversed = compressor.inverse_transform(compressed)
    compression.append((compressor, compressed))
    new_img.append(reversed / 255)
  
  new_img = np.dstack(new_img)
  pickle.dump(compression, open(f'data/{method}_{n_components}.pkl', 'wb'))
  
  return new_img

method = sys.argv[1]
n_com = int(sys.argv[2])
filename = 'ngoclong.jpg'
img = plt.imread(f'data/{filename}')
new_img = compress_image(img, method, n_components=n_com)
plt.imshow(new_img)
plt.title(f'{method} with n_components={n_com}')
plt.savefig(f'data/{method}_{n_com}_{filename}')