from .tree import Tree
import numpy as np
import pandas as pd

class TreeRegression:
    def __init__(self, max_depth=2, min_samples_split=2, criterion='squared_error'):
        if criterion not in ['squared_error', 'absolute_error'] or max_depth <= 0 or min_samples_split <= 0:
            raise ValueError("Invalid input parameters")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = Tree(max_depth)

    def __build_tree(self, X, y, current_node):
        if current_node.current_depth >= self.max_depth:
            current_node.is_leaf = True
            current_node.value = np.mean(y)
            return
        best_feature = None
        best_threshold = None
        current_score = float('inf')
        left_data = right_data = None
        left_y = right_y = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if min(np.sum(left_mask), np.sum(right_mask)) < self.min_samples_split:
                    continue
                if self.criterion == 'squared_error':
                    left_score = self.__mse(y[left_mask])
                    right_score = self.__mse(y[right_mask])
                    score = left_score + right_score
                else:
                    left_score = self.__mae(y[left_mask])
                    right_score = self.__mae(y[right_mask])
                    score = left_score + right_score
                if score < current_score:
                    current_score = score
                    best_feature = feature
                    best_threshold = threshold
                    left_data = X[left_mask]
                    right_data = X[right_mask]
                    left_y = y[left_mask]
                    right_y = y[right_mask]
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
            self.__build_tree(left_data, left_y, current_node.left)
            self.__build_tree(right_data, right_y, current_node.right)

    def fit(self, X, y):
        # Convert to numpy arrays for consistent indexing
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.array(y).flatten()
        self.__build_tree(X, y, self.tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        rezult = np.zeros(len(X))
        for i in range(len(X)):
            rezult[i] = self.predict_one(X[i])
        return rezult

    def predict_one(self, x):
        current_node = self.tree
        while hasattr(current_node, 'is_leaf') and not getattr(current_node, 'is_leaf', False):
            if x[current_node.feature] < current_node.threshold:
                current_node = getattr(current_node, 'left', None)
            else:
                current_node = getattr(current_node, 'right', None)
            if current_node is None:
                break
        return getattr(current_node, 'value', np.nan)

    def score(self, X, y):
        predictions = self.predict(X)
        y_true = np.array(y).flatten()
        if self.criterion == 'squared_error':
            mse = np.mean((predictions - y_true) ** 2)
            y_var = np.var(y_true)
            if y_var == 0:
                return 0.0
            r2 = 1 - (mse / y_var)
            return r2
        elif self.criterion == 'absolute_error':
            mae = np.mean(np.abs(predictions - y_true))
            mean_abs_dev = np.mean(np.abs(y_true - np.mean(y_true)))
            if mean_abs_dev == 0:
                return 0.0
            score = 1 - (mae / mean_abs_dev)
            return score

    def __mse(self, y):
        y_arr = np.array(y).flatten()
        return np.mean((y_arr - np.mean(y_arr)) ** 2)

    def __mae(self, y):
        y_arr = np.array(y).flatten()
        return np.mean(np.abs(y_arr - np.mean(y_arr)))
