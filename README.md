This is a machine learning library developed by Jaden Simon for CS5350/6350 in University of Utah

Decision Tree Library:

Creating a decision tree using this library is done by calling one of three methods within the
DTFactory class. The methods are: dt_entropy (uses entropy for gain), dt_majority (uses majority error),
and dt_gini (uses Gini index). All three take the same parameters. The first parameter is a list of all
examples in the training data set. The second is a list that contains the names of each attribute of an example,
where the last name should be 'label' to mark the output value of the example. The third parameter is a 
list of possible values that each attribute can have. Each element should be a list of values, unless the value
is numeric, then the element should be int. The last parameter is the maximum depth of the tree. Calling any
of the above three functions will return a DecisionTree object. We can then use its 'find_label' method to 
determine the expected label of a given example.

Additional notes:
Since some training or testing data has unknowns or numeric data, it is necessary to preprocess the training
and testing data sets using the 'preprocess_data' method of DTFactory. This will replace numeric fields with
a binary option instead using the median values as a threshold. Addtionally, unknowns will be replaced by
the most common value if the parameter is set. Examples of usage can be found in the hw1_test file.


