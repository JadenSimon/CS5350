This is a machine learning library developed by Jaden Simon for CS5350/6350 at the University of Utah

Decision Tree:

Creating a decision tree using this library is done by calling one of three functions within the
decision tree library. The functions are: dt_entropy (uses entropy for gain), dt_majority (uses majority error),
and dt_gini (uses Gini index). All three take the same parameters. The first parameter is a list of all
examples in the training data set. The second is a list that contains the names of each attribute of an example,
where the last name should be 'label' to mark the output value of the example. The third parameter is a 
list of possible values that each attribute can have. Each element should be a list of values, unless the value
is numeric, then the element should be int. The last parameter is the maximum depth of the tree. Calling any
of the above three functions will return a DecisionTree object. We can then use its 'find_label' method to 
determine the expected label of a given example. 

Currently, this library only supports stumps for weighted decision trees. Use the method 'dt_entropy_stump'
to create a stump from weighted training data. Weights must be given as a list after the examples parameter.
There is no max_depth parameter since the max depth is always 1.

Additional notes:
Since some training or testing data has unknowns or numeric data, it is necessary to preprocess the training
and testing data sets using the 'preprocess_data' method of DTFactory. This will replace numeric fields with
a binary option instead using the median values as a threshold. Addtionally, unknowns will be replaced by
the most common value if the parameter is set. Examples of usage can be found in the hw1_test file.


Ensemble Learning:

Three ensemble classifiers can be created with the following functions: adaboost_DT, bagging_DT, and forest_DT. Each
one has similar parameters with the exception of forest_DT which has an additional parameter that sets the number of
attributes randomly chosen each iteration. Every ensemble classifier method has a number of iterations parameter. This
will set how many trees will be created. All functions return a classifier that can be used to classify examples with the
'find_label' method. The parameter 'print_data' will output additional information every iteration. If this true then
test_data must be passed in as a last parameter in order for error rates to be computed.


Linear Regression:

Two functions are implemented, one is batch_gradient_descent and the other is stochastic_gradient_descent. Both have the same
parameters: a matrix of X values, a vector of y values, the number of iterations to run, a gamma value, a threshold value to know when to stop, and a print_data parameter. Setting print_data shows additional information on every iteration for both
functions. The functions return a weight vector when they either hit the max number of iterations or their weight delta falls
below the threshold.


Perceptron:

The perceptron library contains three different versions: 'standard', voted, and averaged. Each one can be called with paramteters: data, r, T, print_data. Data is simply the matrix of data with x and y values, r is the learning rate, and T is the number of epochs. Setting print_data to True will output additional information every step. All functions return the weight vector found by the algorithm. Note that 1s are automatically appended to the training data to find b. Also note that y is expected to be either 0 or 1. The dual form of perceptron uses a kernel function learn, and returns a error count vector instead
of the normal weight vector.


SVM:

Contains algorithms for learning both primal and dual SVMs. The primal algorithm expects a data matrix, a starting learning rate, a learning rate function that accepts the current learning rate and current epoch as inputs, the max number of epochs, and
the hyperparamter C. It returns a learned weight vector w already containing the bias. The dual version also takes a data matrix,
a kernel function k, the maximum number of epochs, and a hyperparameter C. This function then returns the learned alpha vector. Both functions have a final boolean parameter that prints additional information when set to True. 


Logistic Regression:

Contains logistic_regression (MAP) and logistic_regression_mle. Both take the same parameters and return the found weight vector.
See linear_regression.py for similiar use cases.


Neural Networks:
The neural nets library has two classes: Layer and Network. The Network class is composed of sequential feed-forward layers.
Layers are created with the input and output dimension as well as the activation function and its derivative. Each layer then is
added to the Network in the order desired. No dimension checking is performed, this must be done by the user. To train the network, call the 'train_sgd' method with the desired training dataset. To score the network, use the 'score' method with
any dataset and probability.






