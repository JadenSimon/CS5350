# SVM Library
# Author: Jaden Simon
# Date: 3/24/2019
# Description: Implements various Support Vector Machine algorithms

import numpy as np
from scipy.optimize import minimize

# Performs sub-gradient descent for SVM. r0 is the initial learning rate while r_func is a function used to 
#change the learning rate every epoch
def sgd_svm(data, r0, r_func, T, C, print_data):
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

			if y == 0:
				y = -1

			# Compute the gradient
			grad = w
			if max(0, 1 - y * np.dot(w, x)) != 0:
				grad = grad - (C * y_data.shape[0] * y * x)

			# Update w
			w = w - (r_func(r0, t) * grad)

			# Check for convergence
			if np.linalg.norm(last_grad - grad) < (0.000001 * t):
				return w

			last_grad = grad

		# Print out extra data
		if print_data:
			print "Step " + str(t) + ": w = " + str(w) + " : g = " + str(last_grad)

	return w

# SVM in the dual domain. Allows for kernels. Returns the weights of all data points.
def dual_svm(data, k, T, C, print_data):
	x_data = data[:,:-1]
	y_data = data[:,-1]
	y_data[y_data == 0] = -1 # Convert 0s to -1

	# Initialize our support vectors to all 0s
	a0 = np.zeros(data.shape[0])

	# Create a multiplication factor
	mf = np.zeros((data.shape[0], data.shape[0]))

	for i in range(len(a0)):
		for j in range(len(a0)):
			mf[i][j] = y_data[i] * y_data[j] * k(x_data[j], x_data[i])

	# Define our optimzation function
	def optim_f(a):
		return (0.5 * np.dot(a.T, np.dot(mf, a))) - a.sum()

	# Define the jacobian matrix
	def optim_jac(a):
		return np.dot(a.T, mf) - np.ones_like(a)

	# Define bounds
	bnds = [(0, C)] * len(a0)

	# Define constraints
	#y_t = y_data.transpose()
	#cons = ({'type': 'eq', 'fun': lambda a: np.dot(a, y_data), 'jac': lambda a: y_t})

	# Run the optimization
	res = minimize(optim_f, a0, method='L-BFGS-B', bounds=bnds, jac=optim_jac, options={'maxiter': T, 'disp': print_data})

	return res.x

