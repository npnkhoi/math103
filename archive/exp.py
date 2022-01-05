from matplotlib import pyplot as plt

img = plt.imread('cat.jpg')
print(img.shape)

def compress(data):
  print(data.shape)
  plt.imshow(data)
  plt.show()

compress(img[:, :, 2])