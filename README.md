# CoreMLBench

CoreMLBench is a modular benchmarking suite that demonstrates how two fundamental machine learning algorithms—Linear Regression and Decision Tree—work “under the hood,” and compares their performance to scikit-learn’s implementations. The goal is to illustrate the gap between naïve, from-scratch code and highly optimized library models on both regression and classification tasks.

---

## Key Features

- **Manual implementations**  
  - **Linear Regression**: gradient-descent training with configurable learning rate and iterations  
  - **Decision Tree**: recursive binary splitting based on impurity (Gini/entropy) for both regression and classification

- **Library comparisons**  
  - scikit-learn’s `LinearRegression`, `DecisionTreeRegressor`, and `DecisionTreeClassifier`

- **Two example tasks**  
  1. **Regression**: predict a continuous target (e.g. housing prices)  
  2. **Classification**: predict a binary or multi-class label (e.g. wine quality)

- **Comprehensive evaluation**  
  - **Regression metrics**: MSE, R², training time  
  - **Classification metrics**: accuracy, F1-score, ROC-AUC, training time  
  - **Visualizations**: learning curves, performance vs. model complexity
