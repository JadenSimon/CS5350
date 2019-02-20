# HW2 test code

import sys
sys.path.append('../DecisionTree')
sys.path.append('../EnsembleLearning')

import decision_tree as DT
import ensemble_learning as EL
import profile

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

# Use AdaBoosting
DT.preprocess_data(bank_examples, bank_test, bank_values, False)
ab = EL.adaboost_DT(bank_examples, bank_names, bank_values, 1000, True, bank_test)
bag = EL.bagging_DT(bank_examples, bank_names, bank_values, 1000, True, bank_test)
#print(DT.compute_error(bag, bank_test))
#profile.run('EL.adaboost_DT(bank_examples, bank_names, bank_values, 5, True, bank_test)')