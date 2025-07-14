import numpy as np
import pandas as pd
from .tree import Tree

class TreeClassifier:
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
        if len(X) == 0 or len(y) == 0:
            current_node.is_leaf = True
            current_node.value = 0  
            return
            
        # Check if all samples belong to the same class
        if len(np.unique(y)) == 1:
            current_node.is_leaf = True
            current_node.value = y.iloc[0] if hasattr(y, 'iloc') else y[0]
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
        for feature in X.columns if hasattr(X, 'columns') else range(X.shape[1]):
            col = X[feature] if hasattr(X, 'columns') else X[:, feature]
            for threshold in np.unique(col):
                if hasattr(X, 'columns'):
                    mask = X[feature] < threshold
                    left_data = X[mask]
                    right_data = X[~mask]
                    mask_np = np.array(mask)
                    y_arr = np.asarray(y)
                    left_y = y_arr[mask_np]
                    right_y = y_arr[~mask_np]
                else:
                    mask = col < threshold
                    left_data = X[mask]
                    right_data = X[~mask]
                    left_y = y[mask]
                    right_y = y[~mask]
                if min(len(left_data), len(right_data)) < self.min_samples_split:
                    continue
                if self.criterion == 'gini':
                    left_score = self.__gini(left_y)
                    right_score = self.__gini(right_y)
                    score = (len(left_data) * left_score + len(right_data) * right_score) / len(X)
                elif self.criterion == 'entropy':
                    left_score = self.__entropy(left_y)
                    right_score = self.__entropy(right_y)
                    score = (len(left_data) * left_score + len(right_data) * right_score) / len(X)
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
                    best_left_data = left_data
                    best_right_data = right_data
                    best_left_y = left_y
                    best_right_y = right_y
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
            self.__build_tree(best_left_data, best_left_y, current_node.left)
            self.__build_tree(best_right_data, best_right_y, current_node.right)
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
    def __gini(self, y):
        if hasattr(y, 'value_counts'):
            counts = y.value_counts().values
        else:
            _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def __entropy(self, y):
        if hasattr(y, 'value_counts'):
            counts = y.value_counts().values
        else:
            _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = y.to_numpy() if hasattr(y, 'to_numpy') else y
        return np.mean(y_pred == y_true)
 
    
        