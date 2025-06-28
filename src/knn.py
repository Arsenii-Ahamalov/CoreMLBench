import numpy as np
import pandas as pd
class KNN:
    def __init__(self,neighbours = 5,p = 1,weights = "uniform",task_class = 'c'):
        if neighbours <= 0 or p not in (1,2) or weights not in ("uniform","distance")\
            or task_class not in ('c','r'):
            raise ValueError("Incorrect input data")
        self.neighbours = neighbours
        self.p = p
        self.task_class = task_class
        self.weights = weights
        self.data = None
        self.target = None
    def fit(self,X,y):
        self.data = X.values
        self.target = y.values
    def predict_one(self,x):
        distances = []
        for i in range(len(self.data)):
            if self.p == 1:
                dist = np.sum(np.abs(x - self.data[i]))
            else:
                dist = np.sqrt(np.sum((x - self.data[i]) ** 2))
            distances.append((dist, self.target[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.neighbours]
        k_targets = [neighbor[1] for neighbor in k_nearest]
        if self.task_class == 'c':
            unique, counts = np.unique(k_targets, return_counts=True)
            return unique[np.argmax(counts)]
        else:
            return np.mean(k_targets)
    def __accuracy(self,y,y_true):
        right_count = 0
        for i in range(len(y)):
            if y[i] == y_true[i]:
                right_count += 1
        return right_count / len(y)
    def __mse(self,y,y_true):
        y_arr = np.array(y)
        y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        return np.mean((y_arr - y_true_arr) ** 2)
    def predict(self,X):
        predictions = []
        for i in range(len(X)):
            predictions.append(self.predict_one(X.iloc[i]))
        return predictions
    def score(self,X,y):
        predictions = self.predict(X)
        if self.task_class == 'c':
            score = self.__accuracy(predictions,y)
            print(f"Accuracy score: {score}")
        elif self.task_class == 'r':
            score = self.__mse(predictions,y)
            print(f"MSE score: {score}")
        return score
