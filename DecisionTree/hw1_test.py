# HW1 test code

import decision_tree


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

# Some definition stuff
LATEX_FORMAT = False	  # Set to true allows easy pasting into Latex

car_examples = import_data("car_train.csv")
car_test = import_data("car_test.csv")
car_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
car_values = [["vhigh", "high", "med", "low", "beep"], ["vhigh", "high", "med", "low"], [2, 3, 4, "5more"], [2, 4, "more"], ["small", "med", "big"], ["low", "med", "high"], ["unacc", "acc", "good", "vgood"]]
bank_examples = import_data("bank_train.csv")
bank_test = import_data("bank_test.csv")
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

print("Car Data Set")
run_experiment(car_examples, car_test, car_names, car_values, 6)

print("Bank Data Set (With Unknowns)")
DTFactory.preprocess_data(bank_examples, bank_test, bank_values, False)
run_experiment(bank_examples, bank_test, bank_names, bank_values, 16)

print("Bank Data Set (Without Unknowns)")
DTFactory.preprocess_data(bank_examples, bank_test, bank_values, True)
run_experiment(bank_examples, bank_test, bank_names, bank_values, 16)