import numpy as np
import pandas as pd
from .tree import Tree

class ClassificationTree:
    def __init__(self, max_depth=2, min_samples_split=2,criterion='gini'):
        if criterion not in ['gini','entropy'] or max_depth <= 0 or min_samples_split <= 0:
            raise ValueError("Invalid input parameters")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = Tree(max_depth)
    def fit(self,X,y):
        self.__build_tree(X,y,self.tree)
    def __build_tree(self,X,y,current_node):
        # Check if dataset is empty
        if len(X) == 0 or len(y) == 0:
            current_node.is_leaf = True
            current_node.value = 0  
            return
            
        # Check if all samples belong to the same class
        if len(np.unique(y)) == 1:
            current_node.is_leaf = True
            current_node.value = y.iloc[0]  
            return
            
        if current_node.current_depth >= self.max_depth:
            current_node.is_leaf = True
            current_node.value = np.argmax(np.bincount(y))
            return
            
        if len(X) < self.min_samples_split:
            current_node.is_leaf = True
            current_node.value = np.argmax(np.bincount(y))
            return
            
        best_score = float('inf')
        best_feature = None
        best_threshold = None
        for feature in X.columns:
            for threshold in X[feature].unique():
                left_data = X[X[feature] < threshold]
                right_data = X[X[feature] >= threshold]
                if min(len(left_data), len(right_data)) < self.min_samples_split:
                    continue
                if self.criterion == 'gini':
                    left_score = self.__gini(y[left_data.index])
                    right_score = self.__gini(y[right_data.index])
                    score = (len(left_data) * left_score + len(right_data) * right_score) / len(X)
                elif self.criterion == 'entropy':
                    left_score = self.__entropy(y[left_data.index])
                    right_score = self.__entropy(y[right_data.index])
                    score = (len(left_data) * left_score + len(right_data) * right_score) / len(X)
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            current_node.is_leaf = True
            current_node.value = np.argmax(np.bincount(y))
        else:
            current_node.feature = best_feature
            current_node.threshold = best_threshold
            current_node.is_leaf = False
            current_node.left = Tree(self.max_depth)
            current_node.right = Tree(self.max_depth)
            current_node.left.current_depth = current_node.current_depth + 1
            current_node.right.current_depth = current_node.current_depth + 1
            self.__build_tree(left_data,y[left_data.index],current_node.left)
            self.__build_tree(right_data,y[right_data.index],current_node.right)
    def predict(self,X):
        rezult = np.zeros(len(X))
        for i in range(len(X)):
            rezult[i] = self.predict_one(X.iloc[i])
        return rezult
    def predict_one(self,x):
        current_node = self.tree
        while not current_node.is_leaf:
            if x[current_node.feature] < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.value
    def __gini(self,y):
        return 1 - np.sum((y.value_counts() / len(y)) ** 2)
    def __entropy(self,y):
        return -np.sum((y.value_counts() / len(y)) * np.log2(y.value_counts() / len(y)))
    def score(self,X,y):
        if self.criterion == 'gini':
            score = self.__gini(y)
            print(f"Gini score: {score}")
        elif self.criterion == 'entropy':
            score = self.__entropy(y)
            print(f"Entropy score: {score}")
        return score
 
    
        