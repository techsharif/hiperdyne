"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
from pprint import pprint


#Import the dataset and define the feature as well as the target datasets / columns#
dataset = pd.read_csv('zoo.csv')#Import all columns omitting the fist which consists the names of the animals
#We drop the animal names since this is not a good feature to split the data on
columns = list(dataset.columns.values)
print(columns)
print(set(dataset[columns[0]].tolist()))
print(dataset[columns[1]].dtypes)
from functions import repair_dataset
dataset = repair_dataset(dataset, .2, 4,.4)
target_attribute_name = columns[-1]
total_data = int(len(dataset[target_attribute_name].tolist()) * .8)
continuous_threshold = .2
continuous_ignore = .4
continuous_partition = 4



###########################################################################################################
from functions import entropy
########################################################################################################### 
    
###########################################################################################################
from functions import InfoGain
       
###########################################################################################################
###########################################################################################################
from functions import ID3               
###########################################################################################################
###########################################################################################################
    
    
from functions import predict
        
        
"""
Check the accuracy of our prediction.
The train_test_split function takes the dataset as parameter which should be divided into
a training and a testing set. The test function takes two parameters, which are the testing data as well as the tree model.
"""
###########################################################################################################
###########################################################################################################

from functions import train_test_split

training_data, testing_data = train_test_split(dataset, total_data)


from functions import test

"""
Train the tree, Print the tree and predict the accuracy
"""
tree = ID3(training_data,training_data,training_data.columns[:-1], target_attribute_name)
pprint(tree)
test(testing_data,tree,target_attribute_name)
