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
			t += alpha[i] * sv[i][-1] * k(sv[i][:-1], x)

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

# Get the bias using alpha and sv
def get_bias(alpha, sv, k):
	b = 0

	for i in range(len(alpha)):
		for j in range(len(alpha)):
			b -= alpha[j] * sv[j][-1] * k(sv[i][:-1], sv[j][:-1])

		b += sv[i][-1]

	return b / len(alpha)

print "Problem 3a (Dual SVM)"
k = lambda x, y: np.dot(x, y) # Linear kernel
alpha = SVM.dual_svm(bank_examples, k, 100, 700.0/873, False)
#alpha, sv = get_sv(alpha, bank_examples, 1e-6)
sv = bank_examples
b = get_bias(alpha, sv, k)
print "\tTraining Error: " + str(kernel_error(alpha, sv, b, bank_examples, k) * 100) + "%"
print "\tTesting Error: " + str(kernel_error(alpha, sv, b, bank_test, k) * 100) + "%"

print "Problem 3b (Dual SVM with Gaussian)"
for c in C_Dual:
	for y in y_Dual:
		k = lambda x1, x2: np.exp(-(np.linalg.norm(x1 - x2)**2) / y)
		alpha = SVM.dual_svm(bank_examples, k, 100, c, False)
		sv = bank_examples
		b = get_bias(alpha, sv, k)
		print "C = " + str(c) + " : gamma = " + str(y) + " : SV Count = " + str(count_sv(alpha, sv, 1e-3))
		print "\tTraining Error: " + str(kernel_error(alpha, sv, b, bank_examples, k) * 100) + "%"
		print "\tTesting Error: " + str(kernel_error(alpha, sv, b, bank_test, k) * 100) + "%"


print "Problem 2a (Sub-gradient Descent SVM)"
for c in C:
	w = SVM.sgd_svm(bank_examples, 0.001, r_func1, 100, c, False)
	print "C = " + str(c) + ": w = " + str(w)
	print "\tTraining Error: " + str(PT.avg_prediction_error(w, bank_examples) * 100) + "%"
	print "\tTesting Error: " + str(PT.avg_prediction_error(w, bank_test) * 100) + "%"

print "Problem 2b (Sub-gradient Descent SVM)"
for c in C:
	w = SVM.sgd_svm(bank_examples, 0.001, r_func2, 100, c, False)
	print "C = " + str(c) + ": w = " + str(w)
	print "\tTraining Error: " + str(PT.avg_prediction_error(w, bank_examples) * 100) + "%"
	print "\tTesting Error: " + str(PT.avg_prediction_error(w, bank_test) * 100) + "%"
