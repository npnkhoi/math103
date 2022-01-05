from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib import pyplot as plt
import numpy as np
import sys, pickle

def compress_image(data, method, n_components=5):
  """Compress an image and save to .pkl file. Return the reversed image"""
  new_img = []
  compression = []

  for i in range(3):
    # Create compressor (SVD or PCA)
    if method == 'pca':
      compressor = PCA(n_components=n_components)
    elif method == 'svd':
      compressor = TruncatedSVD(n_components=n_components)

    # Compress and reverse image
    data = img[:, :, i]
    compressor.fit(data)
    compressed = compressor.fit_transform(data)
    reversed = compressor.inverse_transform(compressed)
    
    compression.append((compressor, compressed))
    new_img.append(reversed / 255)
  
  # save to pickle file
  pickle.dump(compression, open(f'data/{method}_{n_components}.pkl', 'wb'))
  
  # return reversed image
  new_img = np.dstack(new_img)
  return new_img


if __name__ == "__main__":
  # Read command-line variables
  method = sys.argv[1]
  n_com = int(sys.argv[2])

  # Processing image
  filename = 'ngoclong.jpg'
  img = plt.imread(f'data/{filename}')
  new_img = compress_image(img, method, n_components=n_com)

  # Show image
  plt.imshow(new_img)
  plt.title(f'{method} with n_components={n_com}')
  plt.savefig(f'data/{method}_{n_com}_{filename}')