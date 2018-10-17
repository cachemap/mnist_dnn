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

	X_train = X_data[0:m_t,:].T; Y_train = Y_data[0:m_t]
	X_dev   = X_data[m_t:,:].T;  Y_dev   = Y_data[m_t:]

	return X_train, X_dev, Y_train, Y_dev

def load_test_dataset():
	X_test = np.load('test.py')

	return X_test

def create_placeholders(n_x, n_y):

	

def initialize_parameters(num_layers, nn_dims):
	'''
	Initializes weights randomly with Xavier initialization and biases with zeros.

	Arguments:
	num_layers - MIGHT BE USELESS (Try len(nn_dims)) - number of layers in DNN
	nn_dims - array that holds the shapes of each layer's weight/bias tensors.

	Returns:
	parameters - Dictionary holding weights and biases for 
	'''
	parameters = {} # Create empty dictionary

	for i in range(1,num_layers+1):
		parameters['W'+i] = tf.get_variable('W'+i, nn_dims[i,0], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
		parameters['b'+i] = tf.get_variable('b'+i, nn_dims[i,1], initializer = tf.zeros_intializer())

	return parameters

def convert_to_one_hot(labels, num_classes):
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

	# Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name="C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    return one_hot

def forward_propagation(X, parameters, num_layers):    
    # Retrieve the parameters from the dictionary "parameters" 
    A = X;
    for i in range(1,num_layers)
   		W = parameters['W'+i]
   		b = parameters['b'+i]
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

def forward_propagation():
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

	# Initialize all the variables
    init = tf.global_variables_initializer()

	with tf.Session() as sess:
		# Run initialization
		sess.run(init)

		# Mini-batch stochastic training loop...
		for epoch in range(num_epochs):

			epoch_cost = 0.0
			num_minibatches = int(m / minibatch_size)



def predict_on_test():
	X_test = load_test_dataset()




X_train, X_dev, Y_train, Y_dev = load_dataset()

# Normalize image vectors
X_train = X_train / 255.0
X_dev   = X_dev / 255.0

# Convert labels to one-hot matrices
Y_train = convert_to_one_hot(Y_train, 10)
Y_dev   = convert_to_one_hot(Y_dev, 10)

# Train model and calculate training/development set accuracies
model(X_train, Y_train, X_dev, Y_dev)








