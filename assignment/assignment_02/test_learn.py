
import pandas as pd
import numpy as np
from pprint import pprint



 
dataset = pd.read_csv('car_evaluation.csv')

columns = list(dataset.columns.values)
# print(columns)
# print(set(dataset[columns[0]].tolist()))
# print(dataset.dtypes)
from functions import repair_dataset
dataset = repair_dataset(dataset, .2, 4,.4)
target_attribute_name = columns[-1]
total_data = int(len(dataset[target_attribute_name].tolist()) * .95)
continuous_threshold = .2
continuous_ignore = .4
continuous_partition = 4
# print(dataset)






# split data
from functions import train_test_split
training_data, testing_data = train_test_split(dataset, total_data)




# create tree
from functions import ID3               
tree = ID3(training_data,training_data,training_data.columns[:-1], target_attribute_name)
# pprint(tree)

# test
from functions import test
test(testing_data,tree,target_attribute_name)
