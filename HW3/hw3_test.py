# HW3 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../Perceptron')

import numpy as np
import decision_tree as DT
import perceptron as PT

np.set_printoptions(precision=3, suppress=True, threshold=np.nan)

bank_examples = np.array(DT.import_data("bank-note/train.csv"), dtype='float64')
bank_test = np.array(DT.import_data("bank-note/test.csv"), dtype='float64')

print "Problem 2a"
w = PT.perceptron(bank_examples, 1.0, 10, False)
print "Standard Perceptron Vector: " + str(w) + "\nTesting Error: " + str(PT.avg_prediction_error(w, bank_test) * 100) + "%"

print "Problem 2b"
w = PT.voted_perceptron(bank_examples, 1.0, 10, False)
print "Voted Perceptron Matrix:\n" + str(w) + "\nTesting Error: " + str(PT.avg_voted_error(w, bank_test) * 100) + "%"

print "Problem 2c"
w = PT.average_perceptron(bank_examples, 1.0, 10, False)
print "Averaged Perceptron Vector:\n" + str(w) + "\nTesting Error: " + str(PT.avg_prediction_error(w, bank_test) * 100) + "%"