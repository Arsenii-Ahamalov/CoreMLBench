from .tree import Tree
import numpy as np
import pandas as pd
class TreeRegression:
    def __init__(self, max_depth=2, min_samples_split=2,criterion='mse'):
        if criterion not in ['mse','mae'] or max_depth <= 0 or min_samples_split <= 0:
            raise ValueError("Invalid input parameters")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = Tree(max_depth)
    def __build_tree(self,X,y,current_node):
        if current_node.current_depth >= self.max_depth:
            current_node.is_leaf = True
            current_node.value = np.mean(y)
            return
        best_feature = None
        best_threshold = None
        current_score = float('inf')
        for feature in X.columns:
            for threshold in X[feature].unique():
                left_data = X[X[feature] < threshold]
                right_data = X[X[feature] >= threshold]
                if min(len(left_data),len(right_data)) < self.min_samples_split:
                    continue
                if self.criterion == 'mse':
                    left_score = self.__mse(y[left_data.index])
                    right_score = self.__mse(y[right_data.index])
                    score = left_score + right_score
                else:
                    left_score = self.__mae(y[left_data.index])
                    right_score = self.__mae(y[right_data.index])
                    score = left_score + right_score
                if score < current_score:
                    current_score = score
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            current_node.is_leaf = True
            current_node.value = np.mean(y)
            return
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
    def fit(self,X,y):
        self.__build_tree(X,y,self.tree)    
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
    def score(self,X,y):
        predictions = self.predict(X)
        if self.criterion == 'mse':
            mse_score = self.__mse(y)   
            print(f"MSE score: {mse_score}")
        elif self.criterion == 'mae':
            mae_score = self.__mae(y)
            print(f"MAE score: {mae_score}")
        return mse_score if self.criterion == 'mse' else mae_score
    def __mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)
    def __mae(self, y):
        return np.mean(np.abs(y - np.mean(y)))
