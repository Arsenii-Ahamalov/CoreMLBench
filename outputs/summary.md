# CoreMLBench Model Benchmark Summary

## Regression Task (Housing Prices)

| Model                        | Test $R^2$ | Best Parameters |
|------------------------------|:----------:|:-----------------------------------------------------------------------------------------------------------------------------------|
| **sklearn LinearRegression** | 0.6718     | {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False, 'tol': 1e-06}                                           |
| **Custom LinearRegression**  | 0.6718     | {'solver': 'normal', 'learning_rate': 0.001, 'n_iters': 100}                                                                       |
| **sklearn KNN**              | 0.1548     | {'n_neighbors': 7, 'p': 2, 'weights': 'distance'}                                                                                  |
| **Custom KNN**               | 0.2308     | {'n_neighbors': 7, 'p': 2, 'weights': 'uniform', 'task_class': 'r'}                                                                |
| **sklearn TreeRegression**   | 0.3753     | {'criterion': 'squared_error', 'max_depth': 5, 'min_samples_split': 2}                                                             |
| **Custom TreeRegression**    | 0.2989     | {'criterion': 'squared_error', 'max_depth': 9, 'min_samples_split': 5}                                                             |

### Discussion
- **Linear Regression:** Both the sklearn and custom implementations achieved nearly identical $R^2$ scores, demonstrating that the custom implementation is correct and competitive for this task. This is expected, as linear regression is a well-understood algorithm with a closed-form solution.
- **KNN Regression:** Both models performed worse than linear regression, with the custom KNN slightly outperforming sklearn's KNN on this dataset. This may be due to differences in default parameters or distance weighting. KNN is sensitive to feature scaling and the choice of $k$.
- **Decision Tree Regression:** The sklearn tree regressor outperformed the custom implementation, likely due to more advanced optimizations and splitting strategies in sklearn. Both tree models underperformed compared to linear regression, possibly due to the limited depth and lack of feature engineering.

## Classification Task (Wine Quality)

| Model                              | Test Accuracy | Best Parameters |
|-------------------------------------|:-------------:|:--------------------------------------------------------------------------|
| **sklearn LogisticRegression**      | 0.5677        | {'C': 1, 'max_iter': 100, 'solver': 'lbfgs'}                              |
| **Custom LogisticRegression**       | 0.5328        | {'learning_rate': 0.01, 'n_iters': 200, 'fit_intercept': True}            |
| **sklearn KNN**                    | 0.6157        | {'n_neighbors': 9, 'p': 2, 'weights': 'distance'}                         |
| **Custom KNN**                     | 0.5415        | {'n_neighbors': 9, 'p': 2, 'weights': 'uniform', 'task_class': 'c'}       |
| **sklearn TreeClassifier**          | 0.5459        | {'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 4}           |
| **Custom TreeClassifier**           | 0.5284        | {'criterion': 'entropy', 'max_depth': 2, 'min_samples_split': 2}           |

### Discussion
- **Logistic Regression:** The sklearn implementation achieved higher accuracy than the custom model, likely due to more robust optimization and regularization. The custom model still performed reasonably well, validating its correctness.
- **KNN Classification:** Sklearn's KNN classifier outperformed the custom version, possibly due to more sophisticated handling of ties, distance metrics, or optimizations. Both models benefited from hyperparameter tuning.
- **Decision Tree Classification:** The sklearn tree classifier again outperformed the custom version, which is expected given sklearn's mature implementation. Both models achieved moderate accuracy, but could likely be improved with better feature engineering or more extensive tuning.

---

## General Notes and Recommendations
- **Room for Improvement:**
  - The results reported here are based on basic preprocessing and a limited grid search for hyperparameters. With more advanced feature engineering, more extensive hyperparameter tuning, and possibly ensembling, it is possible to achieve significantly higher performance on both tasks.
  - The purpose of this project was not to maximize predictive accuracy, but to provide a fair and transparent comparison between custom implementations and sklearn's reference models.
  - All models were evaluated on the same standardized test sets, and care was taken to avoid data leakage.
- **Reproducibility:**
  - All code, models, and predictions are saved in the `models/` and `outputs/` directories for easy reuse and further analysis.
  - For deployment or further research, consider additional data cleaning, feature selection, and more sophisticated model selection strategies.

---

**For more details, see the individual notebook files and the outputs directory for per-sample predictions.**
