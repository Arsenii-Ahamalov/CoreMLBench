import numpy as np
import pandas as pd
class KNN:
    def __init__(self,n_neighbors = 5,p = 1,weights = "uniform",task_class = 'c'):
        if n_neighbors <= 0 or p not in (1,2) or weights not in ("uniform","distance")\
            or task_class not in ('c','r'):
            raise ValueError("Incorrect input data")
        self.n_neighbors = n_neighbors
        self.p = p
        self.task_class = task_class
        self.weights = weights
        self.data = None
        self.target = None
    def fit(self, X, y):
        self.data = X.values if hasattr(X, 'values') else np.array(X)
        self.target = y.values if hasattr(y, 'values') else np.array(y)
    def predict_one(self,x):
        distances = []
        for i in range(len(self.data)):
            if self.p == 1:
                dist = np.sum(np.abs(x - self.data[i]))
            else:
                dist = np.sqrt(np.sum((x - self.data[i]) ** 2))
            distances.append((dist, self.target[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.n_neighbors]
        k_targets = [neighbor[1] for neighbor in k_nearest]
        if self.task_class == 'c':
            unique, counts = np.unique(k_targets, return_counts=True)
            return unique[np.argmax(counts)]
        else:
            return np.mean(k_targets)
    def __accuracy(self, y_pred, y_true):
        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true.iloc[i]:
                correct += 1
        print(f"Correct predictions:")
        return correct / len(y_pred)
    def __mse(self,y_pred,y_true):
        y_arr = np.array(y_pred)
        y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        return np.mean((y_arr - y_true_arr) ** 2)
    def predict(self, X):
        if hasattr(X, 'iloc'):
            return [self.predict_one(X.iloc[i]) for i in range(len(X))]
        else:
            return [self.predict_one(X[i]) for i in range(len(X))]
    def score(self, X, y):
        predictions = self.predict(X)
        if self.task_class == 'c':
            y_true = y.values if hasattr(y, 'values') else np.array(y)
            correct = np.sum(np.array(predictions) == y_true)
            return correct / len(y_true)
        elif self.task_class == 'r':
            y_true = y.values if hasattr(y, 'values') else np.array(y)
            mse = self.__mse(predictions, y_true)
            y_var = np.var(y_true)
            if y_var == 0:
                return 0.0
            r2 = 1 - (mse / y_var)
            return r2
