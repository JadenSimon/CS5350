# Decision Tree Library
# Author: Jaden Simon
# Date: 2/5/2019
# Description: Implements the ID3 algorithm for creating decision trees

import math

# A decision tree. Stores the root node and allows finding labels for a given example.
# Also stores the attribute names in "names"
class DecisionTree:

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



# Decision tree factory class. Creates a new decision tree from the specified parameters.
# Training set should be provided as a list of examples.
class DTFactory:

	#ID3 algorithm
	@staticmethod
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
			return Node(DTFactory.common_value(examples, label_index))

		# Now begin the splitting process by choosing best attribute
		attribute = DTFactory.best_attribute(examples, names, possible_values, gain_function)
		attribute_index = names.index(attribute)
		new_root = Node(attribute)

		# Create new names/values lists without the attribute
		new_names = list(names)
		new_values = list(possible_values)
		new_names[attribute_index] = ""
		new_values[attribute_index] = []

		# If value type is a list (and thus a characteristic) treat it normally
		if isinstance(possible_values[attribute_index], list):
			for value in possible_values[attribute_index]:
				# Create a subset of examples that share the same value
				subset =  DTFactory.shared_values(examples, attribute_index, value)

				new_node = None

				# Empty subset, add leaf for most common label
				if len(subset) == 0:
					new_node = Node(DTFactory.common_value(examples, label_index))
				else:
					new_node = DTFactory.id3(subset, new_names, new_values, gain_function, max_depth - 1)

				new_root.add_child(new_node, value)

		elif possible_values[attribute_index] == int or possible_values[attribute_index] == float:
			# We will choose the median of the numerical values, then generate two subsets
			threshold, subset1, subset2 = DTFactory.split_numerical(examples, attribute_index)

			lt_node = DTFactory.id3(subset1, new_names, new_values, gain_function, max_depth - 1)
			gt_node = DTFactory.id3(subset2, new_names, new_values, gain_function, max_depth - 1)

			new_root.add_child(lt_node, " < " + str(threshold))
			new_root.add_child(gt_node, ">= " + str(threshold))
		else:
			print("Invalid possible value in DTFactory")
			return None

		return new_root

	# Returns the most common value
	@staticmethod
	def common_value(examples, value_index):
		track = {}

		for example in examples:
			value = example[value_index]

			if value not in track:
				track[value] = 1
			else:
				track[value] += 1

		return max(track, key=track.get)

	# Returns the median of a list
	# Assumes it's already sorted
	@staticmethod
	def median(lst):
		n = len(lst)
		half = n//2

		if n < 1: 		
			return None
		if n % 2 == 1: 	
			return lst[half]
		else:
			return sum(lst[half-1:half+1]) / 2.0


	# Returns the best attribute from a list of examples using a gain function
	@staticmethod
	def best_attribute(examples, names, possible_values, gain_function):
		attribute_gains = {}

		for attribute in names:
			if attribute != "label" and attribute != "":
				attribute_index = names.index(attribute)

				# We must iterate over all possible values for the given attribute
				if isinstance(possible_values[attribute_index], list):
					for value in possible_values[attribute_index]:
						subset = DTFactory.shared_values(examples, attribute_index, value)
						proportion = float(len(subset)) / len(examples)

						if attribute not in attribute_gains:
							attribute_gains[attribute] = proportion * gain_function(subset, names.index("label"))
						else:
							attribute_gains[attribute] += proportion * gain_function(subset, names.index("label"))
				elif possible_values[attribute_index] == int or possible_values[attribute_index] == float:
					# If we have a possible value that is numerical, then we can choose to split perfectly on it
					return names[attribute_index]

		return min(attribute_gains, key=attribute_gains.get)

	# Returns the subset of examples that share the same attribute value
	@staticmethod
	def shared_values(examples, attribute_index, value):
		subset = []
		for example in examples:
			if example[attribute_index] == value:
				subset.append(example)

		return subset

	# Splits a set of numerical data in half using its median, returns the threshold as well as the two split sets
	@staticmethod
	def split_numerical(examples, attribute_index):
		numerical_data = []
		for example in examples:
			numerical_data.append(example[attribute_index])
		numerical_data.sort()
		median = DTFactory.median(numerical_data)
		lt_set = []
		gt_set = []

		for example in examples:
			if example[attribute_index] < median:
				lt_set.append(example)
			else:
				gt_set.append(example)

		return median, lt_set, gt_set


	# Creates a new tree using the entropy calculation
	@staticmethod
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

			e = 0
			for count in value_counts.values():
				if count != 0:
					e += -1 * (float(count) / len(examples)) * math.log(float(count) / len(examples), 2)

			return e


		root = DTFactory.id3(examples, names, possible_values, entropy, max_depth)

		return DecisionTree(root, names)


# General Node class for use in a tree data structure
# Children can either be Nodes or Links, in which case
# Links contain additional information on what kind of linkage was made
class Node:
	
	def __init__(self, value):
		self.value = value
		self.parent = None
		self.children = []

	def __eq__(self, other):
		if isinstance(other, Link):
			return self == other.node
		elif isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__

		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	# Converts the node into a text representation
	def __str__(self):
		s = str(self.value) + "\n"
		depth = self.depth()

		for child in self.children:
			s = s + ("\t" * (depth + 1)) +  " \___ "  + child.__str__()

		return s

	def is_leaf(self):
		return not self.children

	def is_root(self):
		return not self.parent

	def add_child(self, node, label):
		node.parent = self

		if label is None:
			self.children.append(node)
		else:
			link = Link(label, node)
			self.children.append(link)

	def remove_child(self, node):
		self.children.remove(node)

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
		for child in self.children:
			if isinstance(child, Link):
				if child.label == value:
					return child.node
			else:
				if child.value == value:
					return child

		return None




# Defines a Link object, used to link Nodes, goes only one way
class Link:

	def __init__(self, label, node):
		self.label = label
		self.node = node

	def __eq__(self, other):
		if isinstance(other, Node):
			return other == self.Node
		elif isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__

		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __str__(self):
		return "[" + str(self.label) + "] " + self.node.__str__() 

	def height(self):
		return self.node.height()


def import_data(name):
	data = []

	with open(name) as f:
		for line in f:
			data.append(line.strip().split(','))

	return data


examples = []
examples.append(["S", "H", "H", "W", "No"])
examples.append(["S", "H", "H", "S", "No"])
examples.append(["O", "H", "H", "W", "Yes"])
examples.append(["R", "M", "H", "W", "Yes"])
examples.append(["R", "C", "N", "W", "Yes"])
examples.append(["R", "C", "N", "S", "No"])
examples.append(["O", "C", "N", "S", "Yes"])
examples.append(["S", "M", "H", "W", "No"])
examples.append(["S", "C", "N", "W", "Yes"])
examples.append(["R", "M", "N", "W", "Yes"])
examples.append(["S", "M", "N", "S", "Yes"])
examples.append(["O", "M", "H", "S", "Yes"])
examples.append(["O", "H", "N", "W", "Yes"])
examples.append(["R", "M", "H", "S", "No"])

names = ["O", "T", "H", "W", "label"]
possible_values = [["S", "O", "R"], ["H", "M", "C"], ["H", "N", "L"], ["S", "W"], ["No", "Yes"]]


examples2 = []
examples2.append(["A", 1, 0])
examples2.append(["A", 2, 0])
examples2.append(["B", 1, 1])
examples2.append(["B", 2, 2])
examples2.append(["C", 1, 2])
examples2.append(["C", 2, 4])
names2 = ["Letter", "Number", "label"]
possible_values2 = [["A", "B", "C"], int, [0, 1, 2]]

car_examples = import_data("car_train.csv")
car_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
car_values = [["vhigh", "high", "med", "low"], ["vhigh", "high", "med", "low"], [2, 3, 4, "5more"], [2, 4, "more"], ["small", "med", "big"], ["low", "med", "high"], ["unacc", "acc", "good", "vgood"]]

dt = DTFactory.dt_entropy(car_examples, car_names, car_values, 16)
print(dt)
