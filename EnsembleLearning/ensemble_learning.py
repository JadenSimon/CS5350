# Ensemble Learning Library
# Author: Jaden Simon
# Date: 2/18/2019
# Description: Uses various ensemble methods of learning

# Allows other libraries to be visible
import sys
sys.path.append('../DecisionTree')

import decision_tree as DT
import math
import random


# Simple class that can be used to find a label from a set of hypotheses and votes
class AdaBoost:

	def __init__(self):
		self.votes = []
		self.hypotheses = []

	def add_hypothesis(self, h, vote):
		self.hypotheses.append(h)
		self.votes.append(vote)

	# Finds a label for an example
	def find_label(self, example, ex_id):
		outputs = {}

		for i in range(len(self.hypotheses)):
			h = self.hypotheses[i]
			output = h.find_label(example, ex_id)

			if output not in outputs:
				outputs[output] = self.votes[i]
			else:
				outputs[output] += self.votes[i]

		return max(outputs, key=outputs.get)

# Simple class that contains a find_label method.
# Uses bagging instead of weighted classifiers.
class Bagging:

	def __init__(self):
		self.hypotheses = []

	def add_hypothesis(self, h):
		self.hypotheses.append(h)

	# Finds a label for an example
	def find_label(self, example, ex_id):
		outputs = {}

		for h in self.hypotheses:
			output = h.find_label(example, ex_id)

			if output not in outputs:
				outputs[output] = 1
			else:
				outputs[output] += 1

		return max(outputs, key=outputs.get)

# Implements the AdaBoost algorithm using decision trees
# Returns an object that can find labels.
# Set print_output to true to show additional information.
def adaboost_DT(examples, names, values, iterations, print_output, test_data):
	# Initialize our weight vector, vote vector, and stump list.
	weights = [1.0 / len(examples)] * len(examples)
	votes = []
	stumps = []
	training_ids = range(len(examples))
	test_ids = range(len(examples), len(test_data) + len(examples))
	ab = AdaBoost()

	if print_output:
		print "Iteration\tTraining Error\tTest Error\tStump Training Error\tStump Test Error"

	# Do a certain amount of iterations
	for t in range(iterations):
		# Create a DT stump using entropy
		stump = DT.dt_entropy_stump(examples, weights, names, values)
		stumps.append(stump)

		# Find its vote
		error = DT.compute_weighted_error(stump, examples, weights, training_ids)
		votes.append(0.5 * math.log((1.0 - error) / error))
		ab.add_hypothesis(stump, votes[t])

		if print_output:
			print "" + str(t + 1) + " \t" + str(DT.compute_error(ab, examples, training_ids)) + " \t" + str(DT.compute_error(ab, test_data, test_ids)) + "\t" + str(DT.compute_error(stump, examples, training_ids)) + " \t" + str(DT.compute_error(stump, test_data, test_ids))

		# Update our weight vector
		for i in range(len(weights)):
			label = stump.find_label(examples[i], training_ids[i])
			if label != examples[i][-1]:
				weights[i] *= math.exp(votes[t])
			else:
				weights[i] *= math.exp(-1 * votes[t])

		# Normalize it
		total = sum(weights)
		for i in range(len(weights)):
			weights[i] /= total

	# Create a new adaboost object to return

	return ab


# Bagging algoirthm using DTs for the number of iterations.
# Set print_output to true to show additional information.
def bagging_DT(examples, names, values, iterations, print_output, test_data):
	trees = []
	bag = Bagging()
	training_ids = range(len(examples))
	test_ids = range(len(examples), len(test_data) + len(examples))

	if print_output:
		print "Tree Count\tTraining Error\t Test Error"

	# Create n trees
	for t in range(iterations):
		# We will create a new training data set from the examples
		# by randomly sampling from it. This means that some examples
		# may get repeated in our new training set.
		training_set = []
		for i in range(len(examples)//4):
			training_set.append(random.choice(examples))

		dt = DT.dt_entropy(training_set, names, values, 100)

		trees.append(dt)
		bag.add_hypothesis(dt)

		if print_output:
			print "" + str(t + 1) + " \t" + str(DT.compute_error(bag, examples, training_ids)) + " \t" + str(DT.compute_error(bag, test_data, test_ids))

	# Return a bagging object

	return bag


# Bagging algoirthm using DTs for the number of iterations.
# Set print_output to true to show additional information.
def forest_DT(examples, names, values, num_attr, iterations, print_output, test_data):
	trees = []
	forest = Bagging()
	training_ids = range(len(examples))
	test_ids = range(len(examples), len(test_data) + len(examples))

	if print_output:
		print "Tree Count\tTraining Error\t Test Error"

	# Create n trees
	for t in range(iterations):
		# We will create a new training data set from the examples
		# by randomly sampling from it. This means that some examples
		# may get repeated in our new training set.
		training_set = []
		for i in range(len(examples)):
			training_set.append(random.choice(examples))

		dt = DT.DecisionTree(DT.id3_random_forest(training_set, names, values, 100, num_attr), names)

		trees.append(dt)
		forest.add_hypothesis(dt)

		if print_output:
			print "" + str(t + 1) + " \t" + str(DT.compute_error(forest, examples, training_ids)) + " \t" + str(DT.compute_error(forest, test_data, test_ids))

	# Return a bagging object
	return forest

