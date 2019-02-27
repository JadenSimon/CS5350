# HW2 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../EnsembleLearning')
sys.path.append('../LinearRegression')

import numpy as np
import decision_tree as DT
import ensemble_learning as EL
import random
import linear_regression as LR

# Load the bank data set
bank_examples = DT.import_data("bank_train.csv")
bank_test = DT.import_data("bank_test.csv")
bank_names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"]
bank_values = []
bank_values.append(int)
bank_values.append(["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", "technician", "services"])
bank_values.append(["married", "divorced", "single"])
bank_values.append(["unknown", "secondary", "primary", "tertiary"])
bank_values.append(["yes", "no"])
bank_values.append(int)
bank_values.append(["yes", "no"])
bank_values.append(["yes", "no"])
bank_values.append(["unknown", "telephone", "cellular"])
bank_values.append(int)
bank_values.append(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
bank_values.append(int)
bank_values.append(int)
bank_values.append(int)
bank_values.append(int)
bank_values.append(["unknown", "other", "failure", "success"])
bank_values.append(["yes", "no"])

DT.preprocess_data(bank_examples, bank_test, bank_values, False)
print "Running AdaBoost Test"
ab = EL.adaboost_DT(bank_examples, bank_names, bank_values, 100, True, bank_test)

print "Running Bagging Test"
bag = EL.bagging_DT(bank_examples, bank_names, bank_values, 100, True, bank_test)

print "Running Forest Test (Feature Size = 2)"
forest = EL.forest_DT(bank_examples, bank_names, bank_values, 2, 100, True, bank_test)

print "Running Forest Test (Feature Size = 4)"
forest = EL.forest_DT(bank_examples, bank_names, bank_values, 4, 100, True, bank_test)

print "Running Forest Test (Feature Size = 6)"
forest = EL.forest_DT(bank_examples, bank_names, bank_values, 6, 100, True, bank_test)

# Dump all the old classifers
ab = None
bag = None
forest = None

# Load the concrete data
concrete_training = np.array(DT.import_data("concrete_train.csv"), dtype='float64')
print(concrete_training.shape[0])
training_x = np.append(concrete_training[:,0:7], np.full((concrete_training.shape[0], 1), 1.0), 1)
training_y = concrete_training[:,7].flatten()
concrete_test = np.array(DT.import_data("concrete_test.csv"), dtype='float64')
test_x = np.append(concrete_test[:,0:7], np.full((concrete_test.shape[0], 1), 1.0), 1)
test_y = concrete_test[:,7].flatten()

# Running batch gradient descent
print "Batch Gradient Descent (Learning Rate = 0.001)"
w = LR.batch_gradient_descent(training_x, training_y, 1000, 0.01, 0.01, True)
print "Test Set MSE: " + str(LR.squared_error(w, test_x, test_y) / concrete_test.shape[0])

# Running stochastic gradient descent
print "Stochastic Gradient Descent (Learning Rate = 0.01)"
w = LR.stochastic_gradient_descent(training_x, training_y, 1000, 0.02, 0.000001, True)
print "Test Set MSE: " + str(LR.squared_error(w, test_x, test_y) / concrete_test.shape[0])

# Solving weight vector analytically
x = np.transpose(training_x)
y = training_y
w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x, np.transpose(x))), x), y)
print 'Computed weight vector: ' + str(w / np.linalg.norm(w))
print "Training Set MSE: " + str(LR.squared_error(np.reshape(w, (8)), training_x, training_y) / concrete_training.shape[0])
print "Test Set MSE: " + str(LR.squared_error(np.reshape(w, (8)), test_x, test_y) / concrete_test.shape[0])  

# Now create 100 Bagging classifiers using 100 uniformly sampled examples without replacement
# We will treat 'yes' as 1 and 'no' as -1
print "Creating 100 Bagging Classifiers (takes a while)"
bags = []
for i in range(100):
	predictions = []

	# First take 1000 random samples and build a classifier
	data = list(bank_examples)
	samples = []
	for j in range(1000):
		index = random.randrange(len(data))
		samples.append(data.pop(index))

	bags.append(EL.bagging_DT(samples, bank_names, bank_values, 100, False, bank_test))

	sys.stdout.write('.')
	sys.stdout.flush()

print "\nComputing Bias and Variance"
bias = 0
variance = 0
bag_bias = 0
bag_variance = 0

for i in range(len(bank_test)):
	test_ex = bank_test[i]
	tree_predictions = np.full(((len(bank_test))), -1, dtype='float')
	bag_predictions = np.full((len(bank_test)), -1, dtype='float')

	for j in range(len(bags)):
		p = bags[j].hypotheses[0].find_label(test_ex, i)
		if p == "yes":
			tree_predictions[j] = 1.0

		p = bags[j].find_label(test_ex, i)
		if p == "yes":
			bag_predictions[j] = 1.0

	y = -1.0
	if test_ex[-1] == "yes":
		y = 1.0

	tree_mean = sum(tree_predictions) / len(tree_predictions)
	bag_mean = sum(bag_predictions) / len(bag_predictions)
	bias += (tree_mean - y)**2
	bag_bias += (bag_mean - y)**2
	variance += sum((tree_predictions - tree_mean)**2) / len(tree_predictions)
	bag_variance += sum((bag_predictions - bag_mean)**2) / len(bag_predictions)

	if i % 50 == 49:
		sys.stdout.write('.')
		sys.stdout.flush()

print "\nBias: " + str(bias / len(bank_test)) + " Variance: " + str(variance / len(bank_test))
print "Bagging Bias: " + str(bag_bias / len(bank_test)) + " Bagging Variance: " + str(bag_variance / len(bank_test))


# Get rid of the created classifiers
bags = []

# Now create 100 Forest classifiers using 100 uniformly sampled examples without replacement
# We will treat 'yes' as 1 and 'no' as -1
print "Creating 100 Forest Classifiers with Feature Size = 2 (takes a while)"
forests = []
for i in range(100):
	predictions = []

	# First take 1000 random samples and build a classifier
	data = list(bank_examples)
	samples = []
	for j in range(1000):
		index = random.randrange(len(data))
		samples.append(data.pop(index))

	forests.append(EL.forest_DT(samples, bank_names, bank_values, 2, 50, False, bank_test))

	sys.stdout.write('.')
	sys.stdout.flush()

print "\nComputing Bias and Variance"
bias = 0
variance = 0
forest_bias = 0
forest_variance = 0

for i in range(len(bank_test)):
	test_ex = bank_test[i]
	tree_predictions = np.full(((len(bank_test))), -1, dtype='float')
	forest_predictions = np.full((len(bank_test)), -1, dtype='float')

	for j in range(len(forests)):
		p = forests[j].hypotheses[0].find_label(test_ex, i)
		if p == "yes":
			tree_predictions[j] = 1.0

		p = forests[j].find_label(test_ex, i)
		if p == "yes":
			forest_predictions[j] = 1.0

	y = -1.0
	if test_ex[-1] == "yes":
		y = 1.0

	tree_mean = sum(tree_predictions) / len(tree_predictions)
	forest_mean = sum(forest_predictions) / len(forest_predictions)
	bias += (tree_mean - y)**2
	forest_bias += (forest_mean - y)**2
	variance += sum((tree_predictions - tree_mean)**2) / len(tree_predictions)
	forest_variance += sum((forest_predictions - forest_mean)**2) / len(forest_predictions)

	if i % 50 == 49:
		sys.stdout.write('.')
		sys.stdout.flush()

print "\nBias: " + str(bias / len(bank_test)) + " Variance: " + str(variance / len(bank_test))
print "Forest Bias: " + str(forest_bias / len(bank_test)) + " Forest Variance: " + str(forest_variance / len(bank_test))