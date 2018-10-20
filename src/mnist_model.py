# Purpose: Construct, train, and test deep NN on MNIST handwritten number
# dataset.

import tensorflow as tf
import numpy as np

# Loads 'X_train.npy' and 'Y_train.npy' to produce training and dev sets
def load_dataset():
	# Load data from file
	data = np.load('../data/train.npy')

	# Randomly shuffle examples to obtain uniform distribution
	m = data.shape[0]       # Total number of examples in data set
	np.random.shuffle(data)

	# Separate data labels from examples
	X_data = data[:,1:]
	Y_data = data[:,0].reshape((m,1))

	# Split dataset into training set (70%) and development set (30%)
	m_t = int(np.ceil(m * 0.7)); print('Number of examples in training set: ', m_t)
	m_d = m - m_t; print('Number of examples in development set: ', m_d)

	X_train = X_data[:m_t,:].T
	Y_train = Y_data[:m_t]
	X_dev   = X_data[m_t:,:].T
	Y_dev   = Y_data[m_t:]

	return X_train, X_dev, Y_train, Y_dev

def load_test_dataset():
	'''
	Load unlabeled dataset to produce predictions for online Kaggle MNIST submission.
	'''
	X_eval = np.load('../data/test.npy')

	return X_eval

def create_placeholders(n_x, n_y):
	'''
	Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x - scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y - scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X - placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y - placeholder for the input labels, of shape [n_y, None] and dtype "float"
	'''

	X = tf.placeholder([n_x, None], dtype=tf.float32, name='X')
	Y = tf.placeholder([n_y, None], dtype=tf.float32, name='Y')

	return X, Y
	

def initialize_parameters(nn_dims):
	'''
	Initializes weights randomly with Xavier initialization and biases with zeros.

	Arguments:
	nn_dims - Dictionary that holds the shapes of each layer's weight/bias tensors.

	Returns:
	parameters - Dictionary holding weights and bias tensors for Tensorflow graph 
	'''

	parameters = {}
	print("len(nn_dims) = " + str(len(nn_dims)//2))

	for i in range(1,len(nn_dims)//2):
		w = 'W'+str(i); b = 'b'+str(i)
		parameters[w] = tf.get_variable(w, nn_dims[w], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
		parameters[b] = tf.get_variable(b, nn_dims[b], initializer=tf.zeros_initializer())

	return parameters

def convert_to_one_hot(labels, num_classes=10):
	'''
	Creates a matrix where the i-th row corresponds to the i-th class number and the j-th column
	    corresponds to the j-th training example. i.e. If example j has label i, entry (i,j) 
	    will be 1 with zeros in all other entries of column j. 

	Arguments:
	labels - vector containing example labels
	num_classes - number of possible classes, "depth" of one-hot matrix

	Returns:
	one_hot - matrix that holds a one in the k-th row of each column
	'''

	# Create a tf.constant equal to C (depth), name it 'C'.
	C = tf.constant(num_classes, name="C")

	# Use tf.one_hot, be careful with the axis 
	one_hot_matrix = tf.one_hot(labels, C, axis=0)

	# Create the session
	sess = tf.Session()

	# Run the session
	one_hot = sess.run(one_hot_matrix)

	# Close the session
	sess.close()

	return one_hot

def forward_propagation(X, parameters, num_layers):    
	A = X;
	for i in range(1,num_layers):
		# Retrieve the parameters from the dictionary "parameters" 
		W = parameters['W'+str(i)]
		b = parameters['b'+str(i)]

		# Z[l+1] = W[l]*A[l] + b[l]
		Z = tf.add(tf.matmul(W,A),b)

		# Calculate activation only if 
		if i is not num_layers:
			A = tf.nn.relu(Z)
	                                                         
	# Return output of last linear unit to compute_cost
	return Z

def compute_cost(logits, labels):
	"""
	Computes the cost

	Arguments:
	Z - output of forward propagation (output of the last LINEAR unit)
	Y - "true" labels vector placeholder, same shape as Z

	Returns:
	cost - Tensor of the cost function
	"""

	# to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
	logits = tf.transpose(logits)
	labels = tf.transpose(labels)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

	return cost

def create_minibatches(X_train, Y_train, minibatch_size):
	None
	

def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
	ops.reset_default_graph()

	n_x = X_train.shape[0]  # Number of pixels in each image / number of input features                
	n_y = Y_train.shape[0]  # Number of classes (10 classes: digits 0-9)
	m   = X_train.shape[1]  # Number of examples                       
	costs = []              # Keep track of model's cost after each epoch for plotting

	# Construct Tensorflow graph
	X, Y = create_placeholders(n_x, n_y)
	parameters = initialize_parameters()
	lin_out = forward_propagation(X, parameters)
	cost = compute_cost(lin_out, Y)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initialize all the variables
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		# Run initialization
		sess.run(init)

		# Form minibatches (random shuffling not needed since dataset is shuffled during load_dataset())
		num_minibatches = int(m / minibatch_size)
		minibatches = create_minibatches(X_train, Y_train, minibatch_size)

		# Mini-batch stochastic training loop...
		for epoch in range(1,num_epochs):
			epoch_cost = 0.0

			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch

				_, minibatch_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

				epoch_cost += minibatch_cost / num_minibatches

			# Print the cost every 100 epochs
			if print_cost and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost and epoch % 10 == 0:
				costs.append(epoch_cost)

	# Plot costs over epochs
	plt.plot(np.squeeze(costs))
	plt.ylabel('Cost')
	plt.xlabel("Iterations (Per 10 Iter.)")
	plt.title("Learning rate =" + str(learning_rate))
	plt.savefig('CostCurve_'+learning_rate)

	# Save trained parameters
	parameters = sess.run(parameters)
	print ("Training complete!")

	# Calculate the correct predictions
	correct_prediction = tf.equal(tf.argmax(lin_out), tf.argmax(Y))

	# Calculate accuracy on the test set
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
	print ("Dev Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}))

	sess.close()

	return parameters


# TODO: DOES THIS WORK PROEPERLY???
def predict_on_test(trained_params):
	# Get unlabeled testing examples
	X_test = load_test_dataset()

	# Normalize image vectors
	X_test = X_test / 255.0

	ops.reset_default_graph()

	n_x = X_test.shape[0]   # Number of pixels in each image / number of input features                                      
	costs = []              # Keep track of model's cost after each epoch for plotting

	# Construct Tensorflow graph
	X, _ = create_placeholders(n_x, 10)
	lin_out = forward_propagation(X_test, parameters)
	predictions = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

	with tf.Session as sess:
		sess.run(predictions)

	sess.close()

	# TODO: Conver this simple print statement to the output specification given by Kaggle
	print(predictions)

# MAIN COMPUTATION DRIVER

# X_train, X_dev, Y_train, Y_dev = load_dataset()

# Normalize image vectors
# X_train = X_train / 255.0
# X_dev   = X_dev / 255.0

# Convert labels to one-hot matrices
# Y_train = convert_to_one_hot(Y_train, 10)
# Y_dev   = convert_to_one_hot(Y_dev, 10)

# Train model and calculate training/development set accuracies
#trained_params = model(X_train, Y_train, X_dev, Y_dev)

# Produce
#predict_on_test(trained_params)

# TEST: Setting up tensorflow graph with layer size specifications
def compute_matrix_dims(layer_sizes):
	'''
	Computes matrix dimensions for each layer

	Arguments:
	layer_sizes - List containing numbers of neurons in each layer

	Asserts:
	n_layers > 2 - Ensures model is at least 2 layers deep

	Returns:
	nn_dims - Dictionary containing tuples of matrix shapes for each layer's weights
		and biases

	'''
	nn_dims = {}
	n_layers = len(layer_sizes)-1

	# Ensure deep learning model
	assert n_layers > 2

	# Build nn_dims list holding matrix sizes
	layer_prev = layer_sizes[0]
	layer_curr = layer_sizes[1]
	for i in range(1,n_layers+1):
		nn_dims["W"+str(i)] = (layer_curr,layer_prev)
		layer_prev = layer_curr
		layer_curr = layer_sizes[i]
		nn_dims['b'+str(i)] = (layer_curr,1)
		
		print("W" + str(i) + " = " + str(nn_dims['W'+str(i)]) + ", "
			+ "b" + str(i) + " = " + str(nn_dims['b'+str(i)]))

	return nn_dims

# TEST:
layer_sizes = [28**2, 12, 12, 10]
nn_dims = compute_matrix_dims(layer_sizes)
parameters = initialize_parameters(nn_dims)

for params in parameters:
	print(params)










