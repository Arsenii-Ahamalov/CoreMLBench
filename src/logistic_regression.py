import numpy as np
import pandas as pd
class LogisticRegression:
    def __init__(self,learning_rate = 0.01,n_iters = 500 ,fit_intercept = True):
        if learning_rate <= 0 or n_iters <= 0:
            raise ValueError("Learning rate and n_iters should be pozitive")
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.bias = 0.0
        self.weights =  None
    def fit(self,X,y):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series)   else y
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iters):
            z = X_mat @ self.weights
            if self.fit_intercept:
                z+=self.bias
            prediction = self.__sigmoid(z)
            error = prediction - y_arr
            grad_w = (X_mat.T @ error) / X.shape[0]
            grad_b = error.sum() / X.shape[0] if self.fit_intercept else 0.0
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
    def predict(self,X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        z = X_mat @ self.weights
        if self.fit_intercept:
            z += self.bias
        prediction = np.empty(X.shape[0],dtype = int)
        for i in range(X.shape[0]):
            prediction[i] = 1 if self.__sigmoid(z[i]) > 0.5 else 0
        return prediction
    def __accuracy(self,y,y_true):
        y_arr   = y_true.values if isinstance(y_true, pd.Series) else y_true
        right_predictions = 0
        for i in range(y.size):
            if y[i] == y_arr[i]:
                right_predictions+=1
        return right_predictions/y.size
    def __sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def score(self,X,y):
        prediction = self.predict(X)
        accuracy_score = self.__accuracy(prediction,y)
        print(f"Model accuracy == {accuracy_score}")
        return accuracy_score
    