# Purpose: Convert provided MNIST handwritten digit dataset to numpy arrays,
# then store these as binaries in NumPy ".npy" format for easier handling
# in TensorFlow.

import numpy as np
import matplotlib.pyplot as plt

def draw_image(arr, label):
	# Use matplotlib to save example as an image
	img = arr.reshape((28,28))
	plt.imshow(img)
	plt.draw()
	plt.savefig('img')
	print("Image label = ",int(label)

# Process "train.csv"
train = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
#test  = np.genfromtxt('test.csv', delimiter=',', skip_header=1)

# Save files to /digit_recognition/data for loading into main Tensorflow project
#np.save('../train.npy', train)
#np.save('../test.npy',  test)

plt.rcParams['image.cmap'] = 'gray' # Set color output to gray
index = 2501
draw_image(train[index,1:(28**2)+1],train[index,0])