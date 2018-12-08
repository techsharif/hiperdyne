import pandas as pd
import numpy as np
from pprint import pprint

def repair_dataset(dataset, threshold=.2, partition=4, ignore=.4):
	for column in list(dataset.columns.values):
		# if len(set(dataset[column].tolist())) / len(dataset[column].tolist()) > ignore:
		# 	dataset=dataset.drop(column,axis=1)
		if len(set(dataset[column].tolist())) / len(dataset[column].tolist()) > threshold:
			if dataset[column].dtypes in [np.float64, np.int64]:
				min_data = min(dataset[column].tolist())
				max_data = max(dataset[column].tolist())
				difference = max_data - min_data
				point = difference / partition
				data_list = []
				for cl in dataset[column]:
					data_list += [int((cl - min_data) // point)]
				dataset[column] = data_list
			else:
				dataset=dataset.drop(column,axis=1)
	return dataset

def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    data = [(-counts[i]/np.sum(counts))*
            np.log2(counts[i]/np.sum(counts)) 
            for i in range(len(elements))
        ]
    entropy = np.sum(data)
    return entropy


def InfoGain(data,split_attribute_name,target_name):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    Weighted_Entropy = np.sum(
            [(counts[i]/np.sum(counts))*
            entropy(
                data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]
            ) 
            for i in range(len(vals))]
        )
    
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data,dataset,features,target_attribute_name="class",parent_node_class = None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data)==0:
        return np.unique(
            dataset[target_attribute_name])[
                np.argmax(np.unique(dataset[target_attribute_name],return_counts=True)[1])
            ]
    elif len(features) ==0:
        return parent_node_class
    else:    
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        
        tree = {best_feature:{}}
        
        features = [i for i in features if i != best_feature]
        
        
        for value in np.unique(data[best_feature]):
            value = value
        
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            tree[best_feature][value] = subtree
            
        return(tree)    

def predict(query,tree,default = 1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


def train_test_split(dataset, total_data):
    training_data = dataset.iloc[:total_data].reset_index(drop=True)
    testing_data = dataset.iloc[total_data:].reset_index(drop=True)
    return training_data,testing_data


def test(data,tree, target_attribute_name):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"]) 
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data[target_attribute_name])/len(data))*100,'%')

