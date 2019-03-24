# HW4 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../Perceptron')
sys.path.append('../SVM')

import numpy as np
import decision_tree as DT
import perceptron as PT
import svm as SVM

np.set_printoptions(precision=3, suppress=True, threshold=np.nan)

bank_examples = np.array(DT.import_data("bank-note/train.csv"), dtype='float64')
bank_test = np.array(DT.import_data("bank-note/test.csv"), dtype='float64')


# Problem 2 stuff
C = [1.0/873, 10.0/873, 50.0/873, 100.0/873, 500.0/873, 700.0/873]

# For problem 2a
d = 10
def r_func1(r, t): 
	return r / (1 + ((r/d) * t)) 

# For problem 2b
def r_func2(r, t):
	return r / (1 + t)

print "Problem 2a (Sub-gradient Descent SVM)"
for c in C:
	w = SVM.perceptron(bank_examples, 1.0, r_func1, 100, c, False)
	print "C = " + str(c) + ": " + str(w)
	print "\tTraining Error: " + str(PT.avg_prediction_error(w, bank_examples * 100) + "%"" Testing Error: " + str(PT.avg_prediction_error(w, bank_test) * 100) + "%""