# Perceptron Library
# Author: Jaden Simon
# Date: 3/4/2019
# Description: Implements Perceptron for learning hyperplane classifiers

import numpy as np 

# Returns the average number of prediction errors
# Appends 1 to the x value
def avg_prediction_error(w, data):
	errors = 0.0

	# Go over every example
	for example in data:
		x = np.append(example[:-1], 1)
		y = example[-1]

		if y == 0:
			y = -1

		if (y * np.dot(w, x)) <= 0:
			errors += 1.0

	return errors / data.shape[0]

# Returns the average number of prediction errors using a weighted perceptron vector
def avg_voted_error(weighted_w, data):
	errors = 0.0

	# Go over every example
	for example in data:
		x = np.append(example[:-1], 1)
		y = example[-1]

		if y == 0:
			y = -1

		# Compute the sum of all weighted w vectors
		s = 0
		for w in weighted_w:
			pred = np.dot(w[0:-1], x)

			if pred <= 0:
				s -= w[-1]
			else:
				s += w[-1]

		if (y * s) <= 0:
			errors += 1.0

	return errors / data.shape[0]



# Standard Unweighted Perceptron
# Runs for T iterations with learning rate r.
# Set print_data for additional information on each iteration.
# Automatically appends a 1 to all x data.
# Returns the w vector.
def perceptron(data, r, T, print_data):
	data = np.array(data) # Make a copy because we will be shuffling it

	w = np.zeros(data.shape[1], dtype='float64')

	# Maximum of T epochs
	for t in range(T):
		np.random.shuffle(data) # Shuffle our training data
		x_data = np.append(data[:,:-1], np.ones((data.shape[0], 1)), axis=1)
		y_data = data[:,-1]

		# Iterate over all examples
		for i in range(y_data.shape[0]):
			x = x_data[i]
			y = y_data[i]

			if y == 0:
				y = -1

			# Check if the weight vector misclassifies the point
			if (y * np.dot(w, x)) <= 0:
				w = w + (r * y * x)

		# Print out extra data
		if print_data:
			print "Step " + str(t) + ": w = " + str(w)

	return w


# Voted Perceptron
# Runs for T iterations with learning rate r.
# Set print_data for additional information on each iteration.
# Automatically appends a 1 to all x data.
# Returns an array of w vectors and weight c scalar.
def voted_perceptron(data, r, T, print_data):
	w = np.zeros(data.shape[1], dtype='float64')
	m = -1
	w_vectors = []
	weights = []

	x_data = np.append(data[:,:-1], np.ones((data.shape[0], 1)), axis=1)
	y_data = data[:,-1]

	# Maximum of T epochs
	for t in range(T):
		# Iterate over all examples
		for i in range(y_data.shape[0]):
			x = x_data[i]
			y = y_data[i]

			if y == 0:
				y = -1

			# Check if the weight vector misclassifies the point
			if (y * np.dot(w, x)) <= 0:
				w = w + (r * y * x)
				m += 1
				weights.append(1)
				w_vectors.append(w)
			else:
				weights[m] += 1


		# Print out extra data
		if print_data:
			print "Step " + str(t) + ": w = " + str(w)

	return np.append(np.array(w_vectors), np.array(weights)[:,None], axis=1)


# Average Perceptron
# Runs for T iterations with learning rate r.
# Set print_data for additional information on each iteration.
# Automatically appends a 1 to all x data.
# Returns an array of w vectors and weight c scalar.
def average_perceptron(data, r, T, print_data):
	w = np.zeros(data.shape[1], dtype='float64')
	a = np.zeros(data.shape[1], dtype='float64')

	x_data = np.append(data[:,:-1], np.ones((data.shape[0], 1)), axis=1)
	y_data = data[:,-1]

	# Maximum of T epochs
	for t in range(T):
		# Iterate over all examples
		for i in range(y_data.shape[0]):
			x = x_data[i]
			y = y_data[i]

			if y == 0:
				y = -1

			# Check if the weight vector misclassifies the point
			if (y * np.dot(w, x)) <= 0:
				w = w + (r * y * x)

			a = a + w

		# Print out extra data
		if print_data:
			print "Step " + str(t) + ": w = " + str(w)

	return a