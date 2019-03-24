# SVM Library
# Author: Jaden Simon
# Date: 3/24/2019
# Description: Implements various Support Vector Machine algorithms

import numpy as np

# Performs sub-gradient descent for SVM. r0 is the initial learning rate while r_func is a function used to 
#change the learning rate every epoch
def sgd_svm(data, r0, r_func, T, C, print_data):
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

			# Compute the gradient
			grad = w
			if max(0, 1 - y * np.dot(w, x)) == 0:
				grad = grad - (C * data.shape[0] * y * x)

			# Update w
			w = w - (r_func(r0, t) * grad)

		# Print out extra data
		if print_data:
			print "Step " + str(t) + ": w = " + str(w)

	return w