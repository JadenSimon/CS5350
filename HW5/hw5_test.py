# HW5 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../LogisticRegression')
sys.path.append('../NeuralNetworks')

import numpy as np
import tensorflow as tf
import decision_tree as DT
import logistic_regression as LR
import neural_nets as NN

from keras.models import Sequential
from keras.layers import Dense

bank_examples = np.array(DT.import_data("bank-note/train.csv"), dtype='float64')
bank_test = np.array(DT.import_data("bank-note/test.csv"), dtype='float64')


# Problem 2 stuff
variance = [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 100.0]

# For problem 2a
d = 0.001
def gamma_func(r, t): 
	return r / (1 + ((r/d) * t)) 

np.set_printoptions(precision=5)

print("Problem 2a (MAP)")

for v in variance:
	w = LR.logistic_regression(bank_examples, 0.01, gamma_func, v, 100, 1e-5)
	train_error = LR.error_rate(bank_examples, w, 0.5) * 100
	test_error = LR.error_rate(bank_test, w, 0.5) * 100
	print("\tv = %3f Train = %5.3f Test = %5.3f " % (v, train_error, test_error), flush=True)

print("Problem 2b (MLE)")

for v in variance:
	w = LR.logistic_regression_mle(bank_examples, 0.01, gamma_func, v, 100, 1e-5)
	train_error = LR.error_rate(bank_examples, w, 0.5) * 100
	test_error = LR.error_rate(bank_test, w, 0.5) * 100
	print("\tv = %3f Train = %5.3f Test = %5.3f " % (v, train_error, test_error), flush=True)


print("Testing neural net on paper problem 3 (problem 3a): ")

layer0 = NN.Layer(3, 2, LR.sigmoid, LR.sigmoid_prime)
layer0.weights = np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
layer1 = NN.Layer(3, 2, LR.sigmoid, LR.sigmoid_prime)
layer1.weights = np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]])
layer2 = NN.Layer(3, 1, lambda x : x, lambda x : np.ones(x.shape))
layer2.weights = np.array([[-1.0], [2.0], [-1.5]])

net = NN.Network()
net.add_layer(layer0)
net.add_layer(layer1)
net.add_layer(layer2)

net.apply_input([1, 1])
gradients = net.backpropagate([1])

print("Layer 2 gradient: \n" + str(np.transpose(gradients[0])))
print("Layer 1 gradient: \n" + str(np.transpose(gradients[1])))
print("Layer 0 gradient: \n" + str(np.transpose(gradients[2])), flush = True)


# Testing training XOR function
#test_data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
#net.train_sgd(test_data, 1.0, lambda x, y: x, 1000)
#print("0 ^ 0 = " + str(net.apply_input([0.0, 0.0])))
#print("0 ^ 1 = " + str(net.apply_input([0.0, 1.0])))
#print("1 ^ 0 = " + str(net.apply_input([1.0, 0.0])))
#print("1 ^ 1 = " + str(net.apply_input([1.0, 1.0])))

print("Problem 3b")
widths = [5, 10, 25, 50, 100]

d = 0.001
def gamma_func(r, t): 
	return r / (1 + ((r/d) * t)) 


feature_count = bank_examples[:,:-1].shape[1]
epochs = 100

for w in widths:
	# Create our layers
	layer0 = NN.Layer(feature_count + 1, w, LR.sigmoid, LR.sigmoid_prime) # input layer
	layer1 = NN.Layer(w + 1, w, LR.sigmoid, LR.sigmoid_prime) # hidden layer 
	layer2 = NN.Layer(w + 1, 1, lambda x : x, lambda x : np.ones(x.shape)) # output layer

	# Create our net
	net = NN.Network()
	net.add_layer(layer0)
	net.add_layer(layer1)
	net.add_layer(layer2)
	net.random_init() # Randomly initialize weights

	# Train our neural net for 100 epochs
	net.train_sgd(bank_examples, 0.1, gamma_func, epochs)
	train_error = net.score(bank_examples, 0.5) * 100
	test_error = net.score(bank_test, 0.5) * 100
	print("\t w = %d Train = %5.3f Test = %5.3f " % (w, train_error, test_error), flush=True)

print("Problem 3c (0 initialized weights)")
for w in widths:
	# Create our layers
	layer0 = NN.Layer(feature_count + 1, w, LR.sigmoid, LR.sigmoid_prime) # input layer
	layer1 = NN.Layer(w + 1, w, LR.sigmoid, LR.sigmoid_prime) # hidden layer 
	layer2 = NN.Layer(w + 1, 1, LR.sigmoid, LR.sigmoid_prime) # output layer

	# Create our net
	net = NN.Network()
	net.add_layer(layer0)
	net.add_layer(layer1)
	net.add_layer(layer2)
	#net.random_init() 

	# Train our neural net for 100 epochs
	net.train_sgd(bank_examples, 0.1, gamma_func, epochs)
	train_error = net.score(bank_examples, 0.5) * 100
	test_error = net.score(bank_test, 0.5) * 100
	print("\t w = %d Train = %5.3f Test = %5.3f " % (w, train_error, test_error), flush=True)

print("Problem 3e (Bonus)")
depths = [3, 5, 9]

# Create, train and score a bunch of TF nets
print ("relu version")
for d in depths:
	for w in widths:
		model = Sequential()
		model.add(Dense(w, kernel_initializer='he_normal', input_dim=feature_count, activation='relu'))

		# Fill the hidden layers
		for i in range(d - 2):
			model.add(Dense(w, kernel_initializer='he_normal', activation='relu'))

		# Make the output layer
		model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(bank_examples[:,:-1], bank_examples[:,-1], epochs=100, batch_size=100, verbose=0)
		train_error = 100.0 - (model.evaluate(bank_examples[:,:-1], bank_examples[:,-1], verbose=0)[1] * 100)
		test_error = 100.0 - (model.evaluate(bank_test[:,:-1], bank_test[:,-1], verbose=0)[1] * 100)

		print("\t d = %d w = %d Train = %5.3f Test = %5.3f " % (d, w, train_error, test_error), flush=True)

# glorot_normal -> xavier initializer
print ("tanh version")
for d in depths:
	for w in widths:
		model = Sequential()
		model.add(Dense(w, kernel_initializer='glorot_normal', input_dim=feature_count, activation='tanh'))

		# Fill the hidden layers
		for i in range(d - 2):
			model.add(Dense(w, kernel_initializer='glorot_normal', activation='tanh'))

		# Make the output layer
		model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))

		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(bank_examples[:,:-1], bank_examples[:,-1], epochs=100, batch_size=100, verbose=0)
		train_error = 100.0 - (model.evaluate(bank_examples[:,:-1], bank_examples[:,-1], verbose=0)[1] * 100)
		test_error = 100.0 - (model.evaluate(bank_test[:,:-1], bank_test[:,-1], verbose=0)[1] * 100)

		print("\t d = %d w = %d Train = %5.3f Test = %5.3f " % (d, w, train_error, test_error), flush=True)

