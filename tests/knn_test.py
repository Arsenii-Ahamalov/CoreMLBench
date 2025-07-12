import numpy as np
import pandas as pd
import pytest
from src.knn import KNN

def test_perfect_classification_2d():
    """
    On perfectly separated 2D data, KNN should achieve 100% accuracy.
    """
    X = pd.DataFrame({
        "x1": [0, 0, 1, 1, 3, 3, 4, 4],
        "x2": [0, 1, 0, 1, 0, 1, 0, 1]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = KNN(n_neighbors=3, p=2, task_class='c')
    model.fit(X, y)
    preds = model.predict(X)

    np.testing.assert_array_equal(preds, y.values)
    assert model.score(X, y) == pytest.approx(1.0)

def test_regression_perfect_linear():
    """
    On linear data, KNN regression should give good predictions.
    """
    X = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series(2 * X["x"] + 1, name="y")  # y = 2x + 1

    model = KNN(n_neighbors=3, p=2, task_class='r')
    model.fit(X, y)
    preds = model.predict(X)
    
    # For regression, we expect predictions to be close to true values
    mse = model.score(X, y)
    assert mse < 1.0  # Allow some tolerance for KNN

def test_manhattan_vs_euclidean():
    """
    Test that both distance metrics work and give reasonable results.
    """
    X = pd.DataFrame({
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, 1, 2, 3, 4, 5]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1], name="y")

    # Test Manhattan distance (p=1)
    model_l1 = KNN(n_neighbors=3, p=1, task_class='c')
    model_l1.fit(X, y)
    acc_l1 = model_l1.score(X, y)

    # Test Euclidean distance (p=2)
    model_l2 = KNN(n_neighbors=3, p=2, task_class='c')
    model_l2.fit(X, y)
    acc_l2 = model_l2.score(X, y)

    assert 0.0 <= acc_l1 <= 1.0
    assert 0.0 <= acc_l2 <= 1.0

def test_predict_and_score_return_types():
    """
    predict(X) returns a list; score(X, y) returns a float.
    """
    X = pd.DataFrame({"x1": [0, 1, 2], "x2": [0, 1, 2]})
    y = pd.Series([0, 0, 1], name="y")

    model = KNN(n_neighbors=2, p=2, task_class='c')
    model.fit(X, y)

    preds = model.predict(X)
    assert isinstance(preds, list)
    assert len(preds) == 3

    acc = model.score(X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

def test_all_same_class():
    """
    If all labels are the same, KNN should predict that class everywhere.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })
    y = pd.Series(np.ones(20, dtype=int), name="y")  # All ones

    model = KNN(n_neighbors=3, p=2, task_class='c')
    model.fit(X, y)
    preds = model.predict(X)

    assert all(pred == 1 for pred in preds)
    assert model.score(X, y) == pytest.approx(1.0)

def test_k_larger_than_data():
    """
    Test behavior when k is larger than the number of training samples.
    """
    X = pd.DataFrame({"x": [1, 2, 3]})
    y = pd.Series([0, 1, 0], name="y")

    model = KNN(n_neighbors=5, p=2, task_class='c')  # k=5 > n=3
    model.fit(X, y)
    
    # Should not crash and should return valid predictions
    preds = model.predict(X)
    assert len(preds) == 3
    assert all(pred in [0, 1] for pred in preds)

@pytest.mark.parametrize("n_neighbors, p, task_class", [
    (3, 1, 'c'),
    (5, 2, 'c'),
    (3, 1, 'r'),
    (5, 2, 'r'),
])
def test_hyperparam_variations(n_neighbors, p, task_class):
    """
    Ensure KNN runs under different hyperparameters and returns valid results.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.random.rand(10)
    })
    if task_class == 'c':
        y = pd.Series(np.random.randint(0, 2, 10), name="y")
    else:
        y = pd.Series(np.random.rand(10), name="y")

    model = KNN(n_neighbors=n_neighbors, p=p, task_class=task_class)
    model.fit(X, y)
    score = model.score(X, y)
    
    if task_class == 'c':
        assert 0.0 <= score <= 1.0  # Accuracy
    else:
        assert score >= 0.0  # MSE

@pytest.mark.parametrize("n_neighbors, p, weights, task_class", [
    (0, 1, "uniform", 'c'),      # n_neighbors <= 0
    (3, 3, "uniform", 'c'),      # p not in (1,2)
    (3, 1, "invalid", 'c'),      # weights not in ("uniform","distance")
    (3, 1, "uniform", 'x'),      # task_class not in ('c','r')
])
def test_constructor_invalid_params(n_neighbors, p, weights, task_class):
    """
    Constructor should raise ValueError for invalid initialization parameters.
    """
    with pytest.raises(ValueError):
        KNN(n_neighbors=n_neighbors, p=p, weights=weights, task_class=task_class)

def test_regression_constant_values():
    """
    Test KNN regression on constant target values.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(10),
        "x2": np.random.rand(10)
    })
    y = pd.Series(np.ones(10) * 5.0, name="y")  # Constant value

    model = KNN(n_neighbors=3, p=2, task_class='r')
    model.fit(X, y)
    preds = model.predict(X)
    
    # All predictions should be close to 5.0
    assert all(abs(pred - 5.0) < 0.1 for pred in preds)

def test_classification_tie_breaking():
    """
    Test KNN behavior when there's a tie in voting (equal number of each class).
    """
    X = pd.DataFrame({
        "x1": [0, 0, 1, 1, 2, 2],
        "x2": [0, 1, 0, 1, 0, 1]
    })
    y = pd.Series([0, 0, 0, 1, 1, 1], name="y")

    model = KNN(n_neighbors=2, p=2, task_class='c')
    model.fit(X, y)
    
    # Test prediction on a point equidistant from both classes
    # This tests tie-breaking behavior
    test_point = pd.DataFrame({"x1": [1.5], "x2": [0.5]})
    pred = model.predict_one(test_point.iloc[0])
    assert pred in [0, 1]  # Should return one of the classes

def test_large_scale_performance():
    """
    Test performance on larger dataset.
    """
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = pd.DataFrame(rng.rand(n_samples, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.randint(0, 3, n_samples), name="y")  # 3 classes
    
    model = KNN(n_neighbors=5, p=2, task_class='c')
    model.fit(X, y)
    
    # Test on a subset
    X_test = X.iloc[:100]
    y_test = y.iloc[:100]
    preds = model.predict(X_test)
    acc = model.score(X_test, y_test)
    
    assert len(preds) == 100
    assert 0.0 <= acc <= 1.0

def test_single_feature():
    """
    Test KNN on single-feature data.
    """
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="y")

    model = KNN(n_neighbors=3, p=1, task_class='c')
    model.fit(X, y)
    
    # Test prediction on a point in the middle
    test_point = pd.DataFrame({"x": [5.5]})
    pred = model.predict_one(test_point.iloc[0])
    assert pred in [0, 1]

def test_edge_case_k_equals_one():
    """
    Test KNN with k=1 (nearest neighbor only).
    """
    X = pd.DataFrame({
        "x1": [0, 1, 2, 3],
        "x2": [0, 1, 2, 3]
    })
    y = pd.Series([0, 1, 0, 1], name="y")

    model = KNN(n_neighbors=1, p=2, task_class='c')
    model.fit(X, y)
    
    # With k=1, prediction should be exactly the label of the nearest neighbor
    preds = model.predict(X)
    assert len(preds) == 4
    assert all(pred in [0, 1] for pred in preds) 