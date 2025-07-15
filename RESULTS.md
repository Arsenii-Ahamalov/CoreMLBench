# CoreMLBench: Final Results & Project Reflection

---

## 🚀 Project Overview

CoreMLBench set out to answer a classic question: **How do machine learning algorithms built from scratch compare to their scikit-learn counterparts?**  
Through hands-on implementation, benchmarking, and careful experiment design, this project provides a transparent, reproducible comparison across both regression and classification tasks.

---

## 📊 What Was Done

- **Implemented from-scratch versions** of Linear Regression, Logistic Regression (with multiclass support), K-Nearest Neighbors, and Decision Trees for both regression and classification.
- **Benchmarked** these custom models against scikit-learn’s highly optimized implementations.
- **Preprocessed data** (cleaning, encoding, scaling) and split into train/validation/test sets to ensure fair evaluation.
- **Tuned hyperparameters** for all models using validation data, keeping the test set strictly for final evaluation.
- **Saved all models and predictions** in organized folders for reproducibility and future use.
- **Tracked experiments** and wrote markdown summaries for each major result.
- **Documented the entire process** in Jupyter notebooks, with code and results.

---

## 🛠️ Features & Workflow

- **Modular codebase:** All custom algorithms are in `src/`, with clear separation from data, models, and outputs.
- **Reproducible experiments:** Notebooks and scripts ensure anyone can rerun the full pipeline.
- **Comprehensive outputs:** Every model’s predictions, metrics, and best parameters are saved for inspection.
- **Markdown reporting:** Both per-experiment (`outputs/summary.md`) and project-wide (`RESULTS.md`) summaries.
- **Extensible design:** Easy to add new algorithms, datasets, or evaluation metrics.

---

## 🔍 Key Findings & Discussion

- **Custom vs. scikit-learn:**  
  - For **Linear Regression**, the custom and sklearn models performed nearly identically, validating the correctness of the from-scratch implementation.
  - For **KNN** and **Decision Trees**, sklearn’s models generally outperformed the custom ones, especially on more complex tasks. This highlights the value of advanced optimizations and engineering in mature libraries.
  - **Logistic Regression** (custom, with multiclass support) was robust and competitive, though sklearn’s version benefited from more sophisticated optimization and regularization.
- **Experiment design matters:**  
  - Strict separation of validation and test data prevented data leakage and ensured honest evaluation.
  - Hyperparameter tuning was performed for both custom and sklearn models, but only on the validation set.
- **Reproducibility:**  
  - All code, models, and results are saved and organized, making it easy to rerun or extend experiments.

---

## ⚠️ Challenges & Lessons Learned


- **Multiclass logic:** Implementing one-vs-rest for custom Logistic Regression was non-trivial but essential for fair comparison.
- **Evaluation metrics:** Making sure custom classifiers returned accuracy (not impurity or entropy) for scoring, to match sklearn’s conventions.
- **Warnings & errors:** Addressing sklearn warnings (e.g., solver deprecation, convergence) and making custom code robust to edge cases.
- **Feature engineering:** The project used basic preprocessing and limited feature engineering; more advanced techniques could further boost performance.

---

## 💡 Conclusions

- **From-scratch ML is educational:** Building algorithms yourself deepens understanding and exposes the subtleties of real-world data science.
- **scikit-learn sets a high bar:** Its models are fast, robust, and highly optimized—custom code can match on simple tasks, but falls behind on complexity and edge cases.
- **Reproducibility and organization matter:** Saving all models, predictions, and experiment logs makes results trustworthy and the project easy to extend.
- **Room for improvement:** With more feature engineering, deeper hyperparameter searches, and ensembling, much higher scores are possible. The focus here was on fair, transparent comparison—not leaderboard-topping accuracy.

---

## 🌱 Next Steps & Recommendations

- Try adding new algorithms (e.g., SVM, ensemble methods) or new datasets.
- Experiment with advanced feature engineering and model selection strategies.
- Use the saved models and predictions for further analysis or deployment.

---

**Thank you for exploring CoreMLBench! For details, see the notebooks, code, and outputs. Contributions and feedback are welcome!**
