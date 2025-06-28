import numpy as np
import pandas as pd
from src.tree_regression_fixed import TreeRegression

# Create simple test data
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
})
y = pd.Series([10, 12, 15, 18, 20, 22, 25, 28])

print("Test Data:")
print("X:")
print(X)
print("\ny:")
print(y)

# Test MSE criterion
print("\n=== Testing MSE Criterion ===")
model_mse = TreeRegression(max_depth=2, min_samples_split=2, criterion='mse')
model_mse.fit(X, y)

predictions_mse = model_mse.predict(X)
print("Predictions (MSE):", predictions_mse)
score_mse = model_mse.score(X, y)
print("Score (MSE):", score_mse)

# Test MAE criterion
print("\n=== Testing MAE Criterion ===")
model_mae = TreeRegression(max_depth=2, min_samples_split=2, criterion='mae')
model_mae.fit(X, y)

predictions_mae = model_mae.predict(X)
print("Predictions (MAE):", predictions_mae)
score_mae = model_mae.score(X, y)
print("Score (MAE):", score_mae)

print("\n=== Test Complete ===") 