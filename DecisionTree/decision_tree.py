# Decision Tree Library
# Author: Jaden Simon
# Date: 2/5/2019
# Description: Implements the ID3 algorithm for creating decision trees

import math

# A decision tree. Stores the root node and allows finding labels for a given example.
# Also stores the attribute names in "names"
class DecisionTree(object):

	__slots__ = ['root', 'names']
	def __init__(self, root, names):
		self.root = root
		self.names = names

	def __str__(self):
		return str(self.root)

	# Determines the label of the given example
	# Note that a node's value refers to the tested attribute if it is not a leaf node
	def find_label(self, example):
		current_node = self.root
		current_attribute = example[self.names.index(current_node.value)]

		if current_node.is_leaf():
			return current_node.value

		while current_attribute is not None:
			current_node = current_node.next_node(current_attribute)

			if not current_node:
				return None
			elif current_node.is_leaf():
				return current_node.value

			current_attribute = example[self.names.index(current_node.value)]

		return None



# Decision tree creation methods.
# Training set should be provided as a list of examples.
# Weights should be a list of floats that describe the weight of each example.

#ID3 algorithm
def id3(examples, names, possible_values, gain_function, max_depth):
	label_index = names.index("label")

	# Check for all same label examples
	same_label = None
	for example in examples:
		if same_label is None:
			same_label = example[label_index]
		elif same_label != example[label_index]:
			same_label = None
			break
	if same_label is not None:
		return Node(same_label)

	# If max depth is reached, or names is exhausted, return leaf with most common label
	if max_depth == 0 or len(list(filter(None, names))) == 1:
		return Node(common_value(examples, label_index))

	# Now begin the splitting process by choosing best attribute
	attribute, split_subsets = best_attribute(examples, names, possible_values, gain_function)
	attribute_index = names.index(attribute)
	new_root = Node(attribute)

	# Create new names/values lists without the attribute
	new_names = list(names)
	new_values = list(possible_values)
	new_names[attribute_index] = ""
	new_values[attribute_index] = []

	# Get the split subsets
	subsets = split_subsets[attribute]

	# If value type is a list (and thus a characteristic) treat it normally
	if type(possible_values[attribute_index]) == list:

		for value in possible_values[attribute_index]:
			# Create a subset of examples that share the same value
			subset = subsets[value]

			new_node = None

			# Empty subset, add leaf for most common label
			if len(subset) == 0:
				new_node = Node(common_value(examples, label_index))
			else:
				new_node = id3(subset, new_names, new_values, gain_function, max_depth - 1)

			new_root.add_child(new_node, value)

	elif type(possible_values[attribute_index]) == float:
		# We will choose the median of the numerical values, then generate two subsets
		threshold = possible_values[attribute_index]
		subset1 = subsets[0]
		subset2 = subsets[1]

		if len(subset1) == 0:
			lt_node = Node(common_value(examples, label_index))
		else:
			lt_node = id3(subset1, new_names, new_values, gain_function, max_depth - 1)

		if len(subset2) == 0:
			gt_node = Node(common_value(examples, label_index))
		else:
			gt_node = id3(subset2, new_names, new_values, gain_function, max_depth - 1)

		new_root.add_child(lt_node, " < " + str(threshold))
		new_root.add_child(gt_node, ">= " + str(threshold))
	else:
		print("Invalid possible value")
		return None

	return new_root

# ID3 algorithm but only with a depth of 1
# Also utilizes weights
def id3_weighted_stump(examples, weights, names, possible_values, gain_function):
	label_index = names.index("label")

	# Now begin the splitting process by choosing best attribute
	attribute, split_subsets, split_weights = best_weighted_attribute(examples, weights, names, possible_values, gain_function)
	attribute_index = names.index(attribute)
	new_root = Node(attribute)

	# Get the split subsets
	subsets = split_subsets[attribute]
	subsets_weights = split_weights[attribute]

	# If value type is a list (and thus a characteristic) treat it normally
	if type(possible_values[attribute_index]) == list:
		for value in possible_values[attribute_index]:
			# Create a subset of examples that share the same value
			subset = subsets[value]
			subset_weights = subsets_weights[value]

			new_node = None

			# Empty subset, add leaf for most common label
			if len(subset) == 0:
				new_node = Node(common_weighted_value(examples, weights, label_index))
			else:
				new_node = Node(common_weighted_value(subset, subset_weights, label_index))

			new_root.add_child(new_node, value)

	elif type(possible_values[attribute_index]) == float:
		# We will choose the median of the numerical values, then generate two subsets
		threshold = possible_values[attribute_index]
		subset1 = subsets[0]
		subset2 = subsets[1]
		subset1_weights = subsets_weights[0]
		subset2_weights = subsets_weights[1]

		if len(subset1) == 0:
			lt_node = Node(common_weighted_value(examples, label_index))
		else:
			lt_node = Node(common_weighted_value(subset1, subset1_weights, label_index))

		if len(subset2) == 0:
			gt_node = Node(common_weighted_value(examples, label_index))
		else:
			gt_node = Node(common_weighted_value(subset2, subset2_weights, label_index))

		new_root.add_child(lt_node, " < " + str(threshold))
		new_root.add_child(gt_node, ">= " + str(threshold))
	else:
		print("Invalid possible value")
		return None

	return new_root

# Returns the most common value
def common_value(examples, value_index):
	track = {}

	for example in examples:
		value = example[value_index]

		if value not in track:
			track[value] = 1
		else:
			track[value] += 1

	return max(track, key=track.get)

# Returns the most common value
def common_weighted_value(examples, weights, value_index):
	track = {}

	for i in range(len(examples)):
		value = examples[i][value_index]

		if value not in track:
			track[value] = weights[i]
		else:
			track[value] += weights[i]

	return max(track, key=track.get)


# Returns the median of a list
def median(lst):
	n = len(lst)
	half = n//2
	lst = list(map(int, lst))
	lst.sort()

	if n < 1: 		
		return None
	if n % 2 == 1: 	
		return lst[half]
	else:
		return sum(lst[half-1:half+1]) / 2.0


# Returns the best attribute from a list of examples using a gain function
# Also returns its split subset.
def best_attribute(examples, names, possible_values, gain_function):
	attribute_gains = {}
	split_subsets = {}

	for attribute in names:
		if attribute != "label" and attribute != "":
			attribute_index = names.index(attribute)

			# We must iterate over all possible values for the given attribute
			if isinstance(possible_values[attribute_index], list):
				subsets = shared_values(examples, attribute_index, possible_values[attribute_index])
				split_subsets[attribute] = subsets

				for value in possible_values[attribute_index]:
					subset = subsets[value]
					proportion = len(subset) / len(examples)

					if attribute not in attribute_gains:
						attribute_gains[attribute] = proportion * gain_function(subset, names.index("label"))
					else:
						attribute_gains[attribute] += proportion * gain_function(subset, names.index("label"))
			elif isinstance(possible_values[attribute_index], float):
				threshold = possible_values[attribute_index]
				subset1, subset2 = split_numerical(examples, attribute_index, threshold)
				split_subsets[attribute] = [subset1, subset2]
				attribute_gains[attribute] = 0
				attribute_gains[attribute] += (len(subset1) / len(examples)) * gain_function(subset1, names.index("label"))
				attribute_gains[attribute] += (len(subset2) / len(examples)) * gain_function(subset2, names.index("label"))

	return min(attribute_gains, key=attribute_gains.get), split_subsets

# Returns the best attribute from a list of examples using a gain function
# Also returns the split subsets.
def best_weighted_attribute(examples, weights, names, possible_values, gain_function):
	attribute_gains = {}
	split_subsets = {}
	split_weights = {}

	for attribute in names:
		if attribute != "label" and attribute != "":
			attribute_index = names.index(attribute)

			# We must iterate over all possible values for the given attribute
			if isinstance(possible_values[attribute_index], list):
				subsets, weighted_subsets = shared_values_weights(examples, weights, attribute_index, possible_values[attribute_index])
				split_subsets[attribute] = subsets
				split_weights[attribute] = weighted_subsets

				for value in possible_values[attribute_index]:
					subset = subsets[value]
					subset_weights = weighted_subsets[value]
					proportion = sum(subset_weights) / sum(weights)

					if attribute not in attribute_gains:
						attribute_gains[attribute] = proportion * gain_function(subset, subset_weights, names.index("label"))
					else:
						attribute_gains[attribute] += proportion * gain_function(subset, subset_weights, names.index("label"))
			elif isinstance(possible_values[attribute_index], float):
				threshold = possible_values[attribute_index]
				subset1, subset2, subset1_weights, subset2_weights = split_numerical_weights(examples, weights, attribute_index, threshold)
				split_subsets[attribute] = [subset1, subset2]
				split_weights[attribute] = [subset1_weights, subset2_weights]
				attribute_gains[attribute] = 0
				attribute_gains[attribute] += (sum(subset1_weights) / sum(weights)) * gain_function(subset1, subset1_weights, names.index("label"))
				attribute_gains[attribute] += (sum(subset2_weights) / sum(weights)) * gain_function(subset2, subset2_weights, names.index("label"))

	return min(attribute_gains, key=attribute_gains.get), split_subsets, split_weights


# Returns a dictionary where each value contains its subset
def shared_values(examples, attribute_index, possible_values):
	subsets = {}

	for value in possible_values:
		subsets[value] = []

	for example in examples:
		subsets[example[attribute_index]].append(example)

	return subsets

# Returns a dictionary where each value contains its subset, also splits the weights
def shared_values_weights(examples, weights, attribute_index, possible_values):
	subsets = {}
	weighted_subsets = {}

	for value in possible_values:
		subsets[value] = []
		weighted_subsets[value] = []

	for i in range(len(examples)):
		value = examples[i][attribute_index]
		subsets[value].append(examples[i])
		weighted_subsets[value].append(weights[i])

	return subsets, weighted_subsets


# Splits a set of numerical data using a threshold
def split_numerical(examples, attribute_index, threshold):
	lt_set = []
	gt_set = []

	for example in examples:
		if float(example[attribute_index]) < threshold:
			lt_set.append(example)
		else:
			gt_set.append(example)

	return lt_set, gt_set


# Splits a set of numerical data using a threshold
def split_numerical_weights(examples, weights, attribute_index, threshold):
	lt_set = []
	gt_set = []
	lt_weights = []
	gt_weights = []

	for i in range(len(examples)):
		if float(examples[i][attribute_index]) < threshold:
			lt_set.append(examples[i])
			lt_weights.append(weights[i])
		else:
			gt_set.append(examples[i])
			gt_weights.append(weights[i])

	return lt_set, gt_set, lt_weights, gt_weights


# Creates a new stump tree using the entropy calculation
def dt_entropy_stump(examples, weights, names, possible_values):

	# Entropy gains function
	def entropy(examples, weights, label_index):
		value_counts = {}

		for i in range(len(examples)):
			value = examples[i][label_index]

			if value not in value_counts:
				value_counts[value] = weights[i]
			else:
				value_counts[value] += weights[i]

		e = 0.0
		for count in value_counts.values():
			if count != 0:
				e += -1 * (float(count) / sum(weights)) * math.log(float(count) / sum(weights), 2)

		return e

	root = id3_weighted_stump(examples, weights, names, possible_values, entropy)

	return DecisionTree(root, names)

# Creates a new tree using the entropy calculation
def dt_entropy(examples, names, possible_values, max_depth):

	# Entropy gains function
	def entropy(examples, label_index):
		value_counts = {}

		for example in examples:
			value = example[label_index]

			if value not in value_counts:
				value_counts[value] = 1
			else:
				value_counts[value] += 1

		e = 0.0
		for count in value_counts.values():
			if count != 0:
				e += -1 * (float(count) / len(examples)) * math.log(float(count) / len(examples), 2)

		return e

	root = id3(examples, names, possible_values, entropy, max_depth)

	return DecisionTree(root, names)

# Creates a new tree using the majority error calculation
def dt_majority(examples, names, possible_values, max_depth):

	# ME gains function
	def majority_error(examples, label_index):
		value_counts = {}

		if len(examples) == 0:
			return 0 

		# Count the label values
		for example in examples:
			value = example[label_index]

			if value not in value_counts:
				value_counts[value] = 1
			else:
				value_counts[value] += 1

		majority_count = max(value_counts.values())

		return 1.0 - (majority_count / len(examples))


	root = id3(examples, names, possible_values, majority_error, max_depth)

	return DecisionTree(root, names)

# Creates a new tree using the gini index calculation
def dt_gini(examples, names, possible_values, max_depth):

	# Gini index gains function
	def gini_index(examples, label_index):
		value_counts = {}

		if len(examples) == 0:
			return 0 

		# Count the label values
		for example in examples:
			value = example[label_index]

			if value not in value_counts:
				value_counts[value] = 1
			else:
				value_counts[value] += 1

		gi = 1.0
		for count in value_counts.values():
			if count != 0:
				gi -= (float(count) / len(examples))**2

		return gi


	root = id3(examples, names, possible_values, gini_index, max_depth)

	return DecisionTree(root, names)

# Does some preprocessing on the possibles values, using the median of the attribute if it is a 
# 'int' value or replacing unknowns if enabled
def preprocess_data(examples, test_data, values, replace_unknowns):
	# Replace all ints with the median of the values
	for i in range(len(values)):
		if values[i] == int:
			num_data = []
			for example in examples:
				num_data.append(example[i])
			values[i] = float(median(num_data))

	# Replace unknowns
	if replace_unknowns:
		# Get a list of most common values
		common_values = []
		for i in range(len(examples[0])):
			common_values.append(common_value(examples, i))

		# Replace unknowns in the training data
		for example in examples:
			for i in range(len(example)):
				if example[i] == "unknown":
					example[i] = common_values[i]

		# Replace unknowns in the testing data
		for example in test_data:
			for i in range(len(example)):
				if example[i] == "unknown":
					example[i] = common_values[i]



# General Node class for use in a tree data structure
# Children can either be Nodes or Links, in which case
# Links contain additional information on what kind of linkage was made
class Node(object):
	
	__slots__ = ['value', 'parent', 'labels', 'children']
	def __init__(self, value):
		self.value = value
		self.parent = None
		self.labels = []
		self.children = []

	# Converts the node into a text representation
	def __str__(self):
		s = str(self.value) + "\n"
		depth = self.depth()

		for child in self.children:
			s = s + ("\t" * (depth + 1)) +  " \___ [" + self.label + "]"  + child.__str__()

		return s

	def is_leaf(self):
		return not self.children

	def is_root(self):
		return not self.parent

	def add_child(self, node, label):
		node.parent = self
		self.children.append(node)
		self.labels.append(label)

	def remove_child(self, node):
		index = self.children.index(node)
		self.children.pop(index)
		self.labels.pop(index)

	# Computes the height of the tree from this node, goes over every child
	def height(self):
		if self.is_leaf():
			return 1
		else:
			greatest_height = 0
			for child in self.children:
				height = child.height()
				if height > greatest_height:
					greatest_height = height

			return greatest_height + 1

	# Computes the depth of the tree from this node
	def depth(self):
		if self.is_root():
			return 0
		else:
			return self.parent.depth() + 1

	# Returns a child that matches either the label or value if it exists
	def next_node(self, value):
		for i in range(len(self.children)):
			if str(self.labels[i]) == value:
				return self.children[i]
			elif str(self.labels[i])[0:3] == " < ":
				if float(value) < float(str(self.labels[i])[3:]):
					return self.children[i]
			elif str(self.labels[i])[0:3] == ">= ":
				if float(value) >= float(str(self.labels[i])[3:]):
					return self.children[i]

		return None

# Imports data into a usable format.
def import_data(name):
	data = []

	# Load the data
	with open(name) as f:
		for line in f:
			data.append(line.strip().split(','))

	return data

# Finds the error rate of a decision tree on test data
def compute_error(dt, test_data):
	errors = 0

	# Iterate over all examples
	for example in test_data:
		label = dt.find_label(example)
		if label != example[-1]:
			errors += 1

	return float(errors) / len(test_data)

# Finds the error rate using weighted data
def compute_weighted_error(dt, test_data, weights):
	errors = 0

	# Iterate over all examples
	for i in range(len(test_data)):
		example = test_data[i]
		label = dt.find_label(example)
		if label != example[-1]:
			errors += weights[i]

	return float(errors) / sum(weights)


# Runs all three types of gain functions on the data from 1 depth to max_depth
def run_experiment(training_data, test_data, names, possible_values, max_depth, latex_format):
	print "Running experiment for max depth of " + str(max_depth) + ":"
	for i in range(max_depth):
		entropy_dt = dt_entropy(training_data, names, possible_values, i + 1)
		majority_dt = dt_majority(training_data, names, possible_values, i + 1)
		gini_dt = dt_gini(training_data, names, possible_values, i + 1)
		trng_entropy_err = round(100 * compute_error(entropy_dt, training_data), 2)
		test_entropy_err = round(100 * compute_error(entropy_dt, test_data), 2)
		trng_majority_err = round(100 * compute_error(majority_dt, training_data), 2)
		test_majority_err = round(100 * compute_error(majority_dt, test_data), 2)
		trng_gini_err = round(100 * compute_error(gini_dt, training_data), 2)
		test_gini_err = round(100 * compute_error(gini_dt, test_data), 2)

		if (i + 1) != entropy_dt.root.height() - 1:
			print "Max tree depth reached!"

		if latex_format == False:
			print "\tDepth " + str(i + 1) + ":" 
			print "\t\t Training: Entropy-> " + str(trng_entropy_err) + " Majority-> " + str(trng_majority_err) + " Gini-> " + str(trng_gini_err)
			print "\t\t Test: Entropy-> " + str(test_entropy_err) + " Majority-> " + str(test_majority_err) + " Gini-> " + str(test_gini_err)
		else:
			print "\t\t" + str(i + 1) + "\t & " + str(trng_entropy_err) + "\%\t & " + str(test_entropy_err) + "\%\t & " + str(trng_majority_err) + "\%\t & " + str(test_majority_err) + "\%\t & " + str(trng_gini_err) + "\%\t & " + str(test_gini_err) + "\%\t \\" + "\\" + " \hline"
