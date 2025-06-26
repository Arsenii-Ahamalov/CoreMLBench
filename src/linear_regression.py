import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate=0.01, solver="normal", n_iters=500):
        if solver not in ("normal", "gd") or n_iters < 0 or learning_rate < 0:
            raise ValueError("Incorrect input data: "
                             "solver must be 'normal' or 'gd', "
                             "n_iters and learning_rate must be non-negative")
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.solver = solver
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X_mat = X.values
        y_vec = y.values
        
        if self.solver == "normal":
            self.fit_normal(X_mat, y_vec)
        else:
            self.fit_gd(X_mat, y_vec)

    def fit_normal(self, X, y):
        X_aug = np.column_stack((X, np.ones(X.shape[0])))
        coefs = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y
        self.weights = coefs[:-1]
        self.bias = coefs[-1]

    def fit_gd(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            predictions = X @ self.weights + self.bias
            errors = predictions - y
            
            grad_w = (2/n_samples) * (X.T @ errors)
            grad_b = (2/n_samples) * errors.sum()
            
            # Update parameters
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, X):
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        return X_mat @ self.weights + self.bias

    def score(self, X, y):
        predictions = self.predict(X)
        return self._mse(predictions, y.values if isinstance(y, pd.Series) else y)

    def _mse(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)