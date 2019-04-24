# Logistic Regression Library
# Author: Jaden Simon
# Date: 4/14/2019
# Description: Implements Logistic Regression for finding MAP or ML estimations

import numpy as np

# Sigmoid function
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))


# Returns the error rate
# Prob is what probability should be used to consider 0 or 1.
def error_rate(data, w, prob):
	count = 0.0

	# Go over every example
	for example in data:
		x = np.append(example[:-1], 1)
		y = example[-1]

		#print(sigmoid(np.dot(x, w)) < prob, y)

		if (sigmoid(np.dot(x, w)) >= prob) != y:
			count += 1.0

	return count / data.shape[0]

# Runs gradient descent with a logistic objective function for T epochs.
# When the gradient is less than the tolerance the function immediately exits.
# r0 is the initial learning rate, r_func is function of the learning rate and number 
# of iterations t, returning a new learning rate for every epoch. v is a hyperparamter
# and is equal to the assumed variance of the features.
# Returns a weight vector w. Automatically appends a 1 to the x_values to find the bias.
def logistic_regression(data, r0, r_func, v, T, tol):
	data = np.array(data) # Make a copy because we will be shuffling it

	w = np.zeros(data.shape[1], dtype='float64')
	last_grad = w

	# Maximum of T epochs
	for t in range(T):
		np.random.shuffle(data) # Shuffle our training data
		x_data = np.append(data[:,:-1], np.ones((data.shape[0], 1)), axis=1)
		y_data = data[:,-1]

		# Iterate over all examples
		for i in range(y_data.shape[0]):
			x = x_data[i]
			y = y_data[i]

			# Compute the gradient
			sig = sigmoid(np.dot(x, w))
			grad = (1.0/v * w) + np.dot(x.T, sig - y)

			# Update w
			w = w - (r_func(r0, t) * grad)
			w /= np.linalg.norm(w)

			# Check for convergence
			if np.linalg.norm(last_grad - grad) < (tol * t):
				return w

			last_grad = grad

	return w

# Same as above but uses MLE instead of MAPE
def logistic_regression_mle(data, r0, r_func, v, T, tol):
	data = np.array(data) # Make a copy because we will be shuffling it

	w = np.zeros(data.shape[1], dtype='float64')
	last_grad = w

	# Maximum of T epochs
	for t in range(T):
		np.random.shuffle(data) # Shuffle our training data
		x_data = np.append(data[:,:-1], np.ones((data.shape[0], 1)), axis=1)
		y_data = data[:,-1]

		# Iterate over all examples
		for i in range(y_data.shape[0]):
			x = x_data[i]
			y = y_data[i]

			# Compute the gradient
			sig = sigmoid(np.dot(x, w))
			grad = np.dot(x.T, sig - y) / v

			# Update w
			w = w - (r_func(r0, t) * grad)
			w /= np.linalg.norm(w)

			# Check for convergence
			if np.linalg.norm(last_grad - grad) < (tol * t):
				return w

			last_grad = grad

	return w