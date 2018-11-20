from mnist_model_deep import *

X_train, X_dev, Y_train, Y_dev = load_dataset()

# Normalize image vectors
X_train = X_train / 255.0
X_dev   = X_dev / 255.0

# Convert labels to one-hot matrices
Y_train = convert_to_one_hot(Y_train, 10)
Y_dev   = convert_to_one_hot(Y_dev, 10)

parameters, errors = model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0001,
	                      num_epochs = 1500, minibatch_size = 32, print_cost = True)
parameters, errors = model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0002,
	                      num_epochs = 1500, minibatch_size = 32, print_cost = True)
parameters, errors = model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0003,
	                      num_epochs = 1500, minibatch_size = 32, print_cost = True)
# Try out different learning rates on the same model
# rates = np.linspace(.0002, .001, 9)
# for l_r in rates:
#	model(X_train, Y_train, X_dev, Y_dev, learning_rate = l_r,
#	      num_epochs = 1500, minibatch_size = 32, print_cost = True)