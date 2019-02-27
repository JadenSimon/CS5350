# Linear Regression Library
# Author: Jaden Simon
# Date: 2/18/2019
# Description: Implements various linear regression algorithms

import numpy as np 
import pandas as pd
from numpy import linalg as npla
import random

# Computes the SSE using a weight vector
def squared_error(w, x, y):
	error = 0
	for i in range(len(y)):
		error = error + (np.dot(w, x[i]) - y[i])**2

	return error

# Finds the gradient at a given point using a weight vector, X vector, and y value.
def compute_gradient(w, x, y):
	return -(y - np.dot(w, x)) * x

# Finds a weight vector for a given X and Y matrices.
# Runs until T iterations or below a given threshold.
# Starts with gamma learning rate.
def batch_gradient_descent(x, y, T, gamma, threshold, print_data):
	w = np.zeros(x.shape[1])

	if print_data:
		print "Iteration\tMSE"

	for t in range(T):
		gradient = np.zeros(x.shape[1])

		# Compute the batch gradient
		for i in range(x.shape[0]):
			gradient += compute_gradient(w, x[i], y[i])

		# Update our weight vector and check if below our threshold
		w -= gamma * gradient
		w /= np.linalg.norm(w)

		# Print out extra data
		if print_data:
			sse = squared_error(w, x, y)
			print "" + str(t) + "\t" + str(sse / x.shape[0])

		if npla.norm(gamma * gradient) < threshold:
			break

	# Print out final data
	if print_data:
		sse = squared_error(w, x, y)
		print "Final:\n\t w = " + str(w)
		print "\tTraining MSE = " + str(sse / x.shape[0])

	return w

# Finds a weight vector for a given X and Y matrices.
# Runs until T iterations or below a given threshold.
# Starts with gamma learning rate.
def stochastic_gradient_descent(x, y, T, gamma, threshold, print_data):
	w = np.zeros(x.shape[1])

	if print_data:
		print "Iteration\tMSE"

	for t in range(T):
		i = random.randrange(x.shape[0])

		# Compute the gradient
		gradient = compute_gradient(w, x[i], y[i])

		# Update our weight vector and check if below our threshold
		w -= gamma * gradient
		w /= np.linalg.norm(w)

		# Print out extra data
		if print_data:
			sse = squared_error(w, x, y)
			print "" + str(t) + "\t" + str(sse / x.shape[0])

		if npla.norm(gamma * gradient) < threshold:
			break

	# Print out final data
	if print_data:
		sse = squared_error(w, x, y)
		print "Final:\n\t w = " + str(w)
		print "\tTraining MSE = " + str(sse / x.shape[0])

	return w

