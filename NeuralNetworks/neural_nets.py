# Neural Networks Library
# Author: Jaden Simon
# Date: 4/14/2019
# Description: Implements simple neural nets. Uses backpropagation to compute the network gradient.

import numpy as np

# Multiple layers
class Network:

	def __init__(self):
		self.layers = [] # Initialize an empty list to store our layers

	# Adds a new layer to the network
	def add_layer(self, layer):
		self.layers.append(layer)

	# Returns the gradient for each layer using a expected output
	def backpropagate(self, y_star, use_bias=True):
		layer_count = len(self.layers)
		output = []
		cached_gradient = []
		last_layer = None

		# We will be going backwards through the layers
		for i in range(layer_count):
			current_layer = self.layers[layer_count - i - 1]
			weight_gradients = np.zeros((current_layer.weights.shape[1], current_layer.weights.shape[0]))

			# If it's the output layer than set cached gradient to the error
			if i == 0:
				cached_gradient = current_layer.output - y_star
			elif use_bias:
				cached_gradient = np.transpose(cached_gradient) * last_layer.gradient * last_layer.weights[1:]
			else:
				cached_gradient = np.transpose(cached_gradient) * last_layer.gradient * last_layer.weights

			gs = np.transpose(cached_gradient) * current_layer.gradient

			for j in range(gs.shape[0]):
				weight_gradients += np.outer(gs[j], current_layer.input)

			output.append(weight_gradients)
			last_layer = current_layer

		return output

	# Applies the input vector, feeding it down the layers to get the output
	# Optionally use use_bias term to append a 1 to each layer
	def apply_input(self, input_vector, use_bias=True):

		# Go over all layers
		for layer in self.layers:

			# Check to see if use bias term
			if use_bias:
				input_vector = np.append([1.0], input_vector)

			# Make the input vector the output of the previous layer
			input_vector = layer.apply_input(input_vector)

		# The final layer is the output
		return input_vector 

	# Randomly initializes all layer weights
	def random_init(self):
		for layer in self.layers:
			layer.random_init()

	# Returns the accuracy of the network on the test data 
	def score(self, data, prob):
		count = 0.0

		# Go over every example
		for example in data:
			x = example[:-1]
			y = example[-1]
			output = self.apply_input(x)[0]
		
			if (output >= prob) != y:
				count += 1.0

		return count / data.shape[0]

	# Trains the neural net using stochastic gradient descent
	# Automatically applies a bias
	def train_sgd(self, data, r0, r_func, T):
		data = np.array(data) # Make a copy because we will be shuffling it

		# Maximum of T epochs
		for t in range(T):
			np.random.shuffle(data) # Shuffle our training data
			x_data = data[:,:-1]
			y_data = data[:,-1]

			for i in range(y_data.shape[0]):
				x = x_data[i]
				y = y_data[i]

				# Propagate the input through the network
				self.apply_input(x)

				# Get the gradients for every node
				gradients = self.backpropagate(y)

				# Update our weights (note that the gradients start at the output layer!)
				for j in range(len(gradients)):
					self.layers[len(gradients) - j - 1].weights -= np.transpose(gradients[j]) * r_func(r0, t)


# A single layer in a neural net
class Layer:

	def __init__(self, input_shape, output_shape, activation_func, derivative_func):
		# Each layer has an input shape and a output shape
		self.weights = np.zeros((input_shape, output_shape), dtype='float64')
		self.gradient = np.zeros((input_shape, output_shape), dtype='float64')
		self.input = np.zeros((input_shape), dtype='float64')
		self.output = np.zeros((output_shape), dtype='float64')

		# Each layer has their own activtion function and its derivative
		self.func = activation_func
		self.func_d = derivative_func

	# Applies a weight vector to a node
	def apply_weight(self, node, weight):
		if weight.shape[0] != self.weights.shape[1]:
			print("Error! Invalid weight vector.")
			return None

		self.weights[node] = weight

	# Applies the weights to an input and uses the activation function
	def apply_input(self, input_vector):
		# Internally store the results
		self.input = input_vector
		self.output = self.func(np.einsum('ij, i->j', self.weights, input_vector))
		self.gradient = self.func_d(np.einsum('ij, i->j', self.weights, input_vector))

		return self.output

	# Randomly assigns the weight values from a normal distribution
	def random_init(self):
		self.weights = np.random.normal(size=self.weights.shape)
