from mnist_model_deep import *

X_train, X_dev, Y_train, Y_dev = load_dataset()

# Normalize image vectors
X_train = X_train / 255.0
X_dev   = X_dev / 255.0

# Convert labels to one-hot matrices
Y_train = convert_to_one_hot(Y_train, 10)
Y_dev   = convert_to_one_hot(Y_dev, 10)

# Train model and calculate training/development set accuracies
trained_params = model(X_train, Y_train, X_dev, Y_dev)

# TODO: Save trained_parameters to file for easy reuse
np.save('./trained_parameters.npy', trained_params)

print(trained_params) # DEBUG: investigate what this looks like

# Produce predictions for Kaggle evaluation
predict_on_test(trained_params)