# HW4 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../Perceptron')
sys.path.append('../SVM')

import numpy as np
import decision_tree as DT
import perceptron as PT
import SVM as SVM

np.set_printoptions(precision=3, suppress=True, threshold=np.nan)

bank_examples = np.array(DT.import_data("bank-note/train.csv"), dtype='float64')
bank_test = np.array(DT.import_data("bank-note/test.csv"), dtype='float64')


# Problem 2 stuff
C = [1.0/873, 10.0/873, 50.0/873, 100.0/873, 500.0/873, 700.0/873]
C_Dual = [100.0/873, 500.0/873, 700.0/873]
y_Dual = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

# For problem 2a
d = 0.001
def r_func1(r, t): 
	return r / (1 + ((r/d) * t)) 

# For problem 2b
def r_func2(r, t):
	return r / (1 + t)

# Returns the average number of prediction errors using a kernel
def kernel_error(alphs, sv, b, data, k):
	errors = 0.0

	# Go over every example
	for example in data:
		x = example[:-1]
		y = example[-1]

		if y == 0:
			y = -1

		t = 0
		for i in range(len(sv)):
			y2 = sv[i][-1]

			if y2 == 0:
				y2 = -1

			t += alpha[i] * y2 * k(sv[i][:-1], x)

		if y * (t + b) <= 0:
			errors += 1.0

	return errors / data.shape[0]

# Returns a list of support vectors from alpha
def get_sv(alpha, data, tol):
	sv = np.array(data)
	new_alpha = np.array(alpha)
	for i in range(len(alpha)):
		if alpha[i] < tol:
			sv = np.delete(sv, data[i], axis=1)
			new_alpha = np.delete(new_alpha, alpha[i])

	return new_alpha, sv

def count_sv(alpha, sv, tol):
	c = 0
	for i in range(len(alpha)):
		if alpha[i] > tol:
			c += 1

	return c

# Get the weight vector using alpha and sv
# w = sum_i alpha_i * y_i * x_i
def get_weight(alpha, sv):
	w = np.zeros(sv.shape[1] - 1, dtype='float64')

	for i in range(len(alpha)):
		w += alpha[i] * sv[i][-1] * sv[i][:-1]

	return w

# Get the bias using alpha and sv
def get_bias(alpha, sv, k):
	b = 0

	for i in range(len(alpha)):
		for j in range(len(alpha)):
			y = sv[j][-1]

			if y == 0:
				y = -1

			b -= alpha[j] * y * k(sv[i][:-1], sv[j][:-1])

		y = sv[i][-1]
		if y == 0:
			y = -1

		b += y

	return b / len(alpha)

# Counts the number of overlappings svs between two alphas
def overlap_svs(alpha1, alpha2, tol):
	c = 0

	for i in range(len(alpha1)):
		if alpha1[i] > tol and alpha2[i] > tol:
			c += 1

	return c

# Problem 2c stuff
delta_w = []
delta_train = []
delta_test = []

print "Problem 2a (Sub-gradient Descent SVM)"
for c in C:
	w = SVM.sgd_svm(bank_examples, 0.001, r_func1, 100, c, False)
	w = w / np.linalg.norm(w)
	train_error = PT.avg_prediction_error(w, bank_examples) * 100
	test_error = PT.avg_prediction_error(w, bank_test) * 100
	delta_w.append(w)
	delta_train.append(train_error)
	delta_test.append(test_error)
	print "C = " + str(c) + ": w = " + str(w)
	print "\tTraining Error: " + str(train_error) + "%"
	print "\tTesting Error: " + str(test_error) + "%"

print "Problem 2b (Sub-gradient Descent SVM)"
for i in range(len(C)):
	c = C[i]
	w = SVM.sgd_svm(bank_examples, 0.001, r_func2, 100, c, False)
	w = w / np.linalg.norm(w)
	train_error = PT.avg_prediction_error(w, bank_examples) * 100
	test_error = PT.avg_prediction_error(w, bank_test) * 100
	delta_w[i] = delta_w[i] - w
	delta_train[i] = delta_train[i] - train_error
	delta_test[i] = delta_test[i] - test_error
	print "C = " + str(c) + ": w = " + str(w)
	print "\tTraining Error: " + str(train_error) + "%"
	print "\tTesting Error: " + str(test_error) + "%"

print "Problem 2c (Model Parameter Difference)"
for i in range(len(C)):
	print "C = " + str(C[i])
	print "\tdelta_w = " + str(delta_w[i])
	print "\tdelta_train = " + str(delta_train[i]) + "%"
	print "\tdelta_test = " + str(delta_test[i]) + "%"

print "Problem 3a (Dual SVM)"
k = lambda x, y: np.dot(x, y) # Linear kernel
for c in C_Dual:
	alpha = SVM.dual_svm(bank_examples, k, 100, c, False)
	sv = bank_examples
	w = get_weight(alpha, sv)
	b = get_bias(alpha, sv, k)
	w = np.append(w, b)
	print "C = " + str(c) + ": w = " + str(w / np.linalg.norm(w))
	print "\tTraining Error: " + str(kernel_error(alpha, sv, b, bank_examples, k) * 100) + "%"
	print "\tTesting Error: " + str(kernel_error(alpha, sv, b, bank_test, k) * 100) + "%"

print "Problem 3b (Dual SVM with Gaussian)"
for c in C_Dual:
	last_alpha = None
	for y in y_Dual:
		k = lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2)**2) / y)
		alpha = SVM.dual_svm(bank_examples, k, 100, c, False)
		sv = bank_examples
		b = get_bias(alpha, sv, k)
		print "C = " + str(c) + " : gamma = " + str(y) + " : SV Count = " + str(count_sv(alpha, sv, 1e-3))
		if last_alpha is not None:
			print "\tOverlapping SVs: " + str(overlap_svs(alpha, last_alpha, 1e-3))
		else:
			print "\tOverlapping SVs: NA"
		print "\tTraining Error: " + str(kernel_error(alpha, sv, b, bank_examples, k) * 100) + "%"
		print "\tTesting Error: " + str(kernel_error(alpha, sv, b, bank_test, k) * 100) + "%"
		last_alpha = alpha

print "Problem 3d (Kernel Perceptron)"
for y in y_Dual:
	k = lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2)**2) / y)
	alpha = PT.dual_perceptron(bank_examples, k, 25, False)
	sv = bank_examples
	b = get_bias(alpha, sv, k)
	print "Gamma = " + str(y) + ":"
	print "\tTraining Error: " + str(kernel_error(alpha, sv, b, bank_examples, k) * 100) + "%"
	print "\tTesting Error: " + str(kernel_error(alpha, sv, b, bank_test, k) * 100) + "%"