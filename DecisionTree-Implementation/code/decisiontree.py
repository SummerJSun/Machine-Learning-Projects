#Collaboration statement: The following materials are used to generate this code:
#lecture material, course google colab 
#psuedocode from: https://www.cs.cmu.edu/~bhiksha/courses/10-601/decisiontrees/
#code from: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#my past code from CS334 machine learning
#stack overflow for debugging
#Sklean packages and my past code from CS334 for evaluation process were used to evaluate the model

import pandas as pd
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=6):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None


    def calculate_entropy(self,value_counts):
        if len(value_counts) == 0:
            return 0
        total = sum(value_counts)
        probs = [count / total for count in value_counts]
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def information_gain(self,parent, left_child, right_child):
        if len(parent) == 0 or len(left_child) == 0 or len(right_child) == 0:
            return 0
        before_split = self.calculate_entropy(parent.value_counts())
        left_split = (len(left_child)/len(parent)) * self.calculate_entropy(left_child.value_counts())
        right_split = (len(right_child)/len(parent)) * self.calculate_entropy(right_child.value_counts())
        after_split = left_split + right_split
        return before_split - after_split

    def split(self,x_train,y_train,feature,threshold):
        if pd.api.types.is_numeric_dtype(x_train[feature]):
            left_x = x_train[feature] <= threshold
            right_x = x_train[feature] > threshold
        else:
            left_x = x_train[feature] == threshold
            right_x = x_train[feature] != threshold
        
        left_y = y_train[left_x]
        right_y = y_train[right_x]
        return left_x, right_x, left_y, right_y

    def find_best_split(self,x_train,y_train):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        if len(y_train) < self.min_samples_leaf:
            return None, None

        for feature in x_train.columns:
            unique_val = x_train[feature].unique()
            
            if len(unique_val) <= 1:
                continue

            if pd.api.types.is_numeric_dtype(x_train[feature]):
                unique_val = sorted(unique_val)
                thresholds = [(unique_val[i] + unique_val[i+1])/2 
                             for i in range(len(unique_val)-1)]
            else:
                thresholds = unique_val

            for threshold in thresholds:
                left_x, right_x, left_y, right_y = self.split(x_train,y_train,feature,threshold)
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = self.information_gain(y_train,left_y,right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self,x_train,y_train,depth=0):
        if len(y_train) == 0:
            return Node(prediction='No')
            
        if len(y_train.unique()) == 1:
            return Node(prediction=y_train.iloc[0])
        
        if depth >= self.max_depth:
            return Node(prediction=y_train.value_counts().idxmax())
    
        best_feature, best_threshold = self.find_best_split(x_train,y_train)
    
        if best_feature is None:
            return Node(prediction=y_train.value_counts().idxmax())
    
        left_x, right_x, left_y, right_y = self.split(x_train,y_train,best_feature,best_threshold)
        
        if len(left_y) == 0:
            return Node(prediction=right_y.value_counts().idxmax())
        if len(right_y) == 0:
            return Node(prediction=left_y.value_counts().idxmax())
            
        node = Node(best_feature,best_threshold)
        node.left = self.build_tree(x_train[left_x],left_y,depth+1)
        node.right = self.build_tree(x_train[right_x],right_y,depth+1)
    
        return node

    def fit(self,x_train,y_train):
        self.root = self.build_tree(x_train,y_train)

    def predict(self,x_test):
        predictions = []
        for _, row in x_test.iterrows():
            node = self.root

            while node.prediction is None:
                if pd.api.types.is_numeric_dtype(pd.Series([row[node.feature]])):
                    if row[node.feature] <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    if row[node.feature] == node.threshold:
                        node = node.left
                    else:
                        node = node.right
            predictions.append(node.prediction)
        return predictions

    def decision_tree(self, path_data, path_training_set, path_test_set, path_output):
        data = pd.read_csv(path_data)
        data["person ID"] = data["person ID"].astype(int)
        
        with open(path_training_set, 'r') as f:
            train_id = [int(line.strip()) for line in f.readlines()]
        with open(path_test_set, 'r') as f:
            test_id = [int(line.strip()) for line in f.readlines()]
        
        train_data = data[data["person ID"].isin(train_id)]
        test_data = data[data["person ID"].isin(test_id)]
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("No matching records found in data for train/test IDs")
        
        x_train = train_data.drop(columns=["person ID","Has heart disease? (Prediction Target)"])
        x_test = test_data.drop(columns=["person ID","Has heart disease? (Prediction Target)"])
        y_train = train_data["Has heart disease? (Prediction Target)"]
        y_test = test_data["Has heart disease? (Prediction Target)"]

        self.fit(x_train,y_train)
        predictions = self.predict(x_test)

        with open(path_output, 'w') as f:
            for pid, pred in zip(test_id, predictions):
                f.write(f"{pid}\t{pred}\n") 







