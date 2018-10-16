# Purpose: Construct, train, and test deep NN on MNIST handwritten number
# dataset.

import tensorflow as tf
import numpy as np

# Loads 'X_train.npy' and 'Y_train.npy' to produce training and dev sets
def load_dataset():
	# Load data from file
	data = np.load('train.npy')

	# Randomly shuffle examples to obtain uniform distribution
	m = data.shape[0]       # Total number of examples in data set
	np.random.shuffle(data)

	# Separate data labels from examples
	X_data = data[:,1:]
	Y_data = data[:,0].reshape((m,1))

	# Split dataset into training set (70%) and development set (30%)
	m_t = np.floor(m * 0.7); print('Number of examples in training set: ', m_t)

	X_train = data[0:m_t,:]; Y_train = 
	X_dev   = X_data[m_t:,:]

	return X_train, X_dev, Y_train, Y_dev

def initialize_parameters(nn_dims):
	None

def forward_propagation():
	None

def optimize_parameters():
	None

def model():
	None




X_train, X_dev, Y_train, Y_dev = load_dataset()