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
			print "Step " + str(t) + ": w = " + str(w)

	return w

# SVM in the dual domain. Allows for kernels. Returns the weights of all data points.
def dual_svm(data, k, T, C, print_data):
	x_data = data[:,:-1]
	y_data = data[:,-1]
	y_data[y_data == 0] = -1 # Convert 0s to -1

	# Randomly initialize weights
	#a0 = np.random.random_sample(data.shape[0]) * C
	a0 = np.zeros(data.shape[0])

	# Create a multiplication factor
	mf = np.zeros((data.shape[0], data.shape[0]))

	for i in range(len(a0)):
		for j in range(len(a0)):
			mf[i][j] = y_data[i] * y_data[j] * k(x_data[i], x_data[j])

	# Define our optimzation function
	def optim_f(a):
		y = 0
		total_a = 0

		for i in range(len(a)):
			total_a += a[i]

			for j in range(len(a)):
				y += mf[i][j] * a[i] * a[j]

		return (0.5 * y) - total_a

	# Define the jacobian matrix
	def optim_jac(a):
		jac = np.zeros(data.shape[0])

		for i in range(len(jac)):
			for j in range(len(jac)):
				if i != j:
					jac[i] += (0.5) * mf[i][j] * a[j]

			jac[i] += (a[i] * mf[i][i]) - 1

		return jac

	# Define bounds
	bnds = [(0, C)] * len(a0)

	# Define constraints
	cons = ({'type': 'eq', 'fun': lambda a: np.dot(a, y_data)})

	# Run the optimization
	res = minimize(optim_f, a0, method='SLSQP', bounds=bnds, constraints=cons, jac=optim_jac, tol=1e-3, options={'maxiter': T, 'disp': print_data})

	return res.x

