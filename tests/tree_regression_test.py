import numpy as np
import pandas as pd
import pytest
from src.tree_regression_fixed import TreeRegression

def test_mse_vs_mae_criteria():
    """
    Test that both MSE and MAE criteria work and give reasonable results.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(50),
        "x2": np.random.rand(50)
    })
    y = pd.Series(3 * X["x1"] + 2 * X["x2"] + np.random.normal(0, 0.1, 50), name="y")

    # Test MSE criterion
    model_mse = TreeRegression(max_depth=3, min_samples_split=2, criterion='mse')
    model_mse.fit(X, y)
    mse_score = model_mse.score(X, y)

    # Test MAE criterion
    model_mae = TreeRegression(max_depth=3, min_samples_split=2, criterion='mae')
    model_mae.fit(X, y)
    mae_score = model_mae.score(X, y)

    assert mse_score >= 0.0
    assert mae_score >= 0.0

def test_predict_and_score_return_types():
    """
    predict(X) returns numpy array; score(X, y) returns float.
    """
    X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    y = pd.Series([10, 20, 30], name="y")

    model = TreeRegression(max_depth=2, min_samples_split=2, criterion='mse')
    model.fit(X, y)

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)

    score = model.score(X, y)
    assert isinstance(score, float)
    assert score >= 0.0

def test_constant_target_values():
    """
    If all target values are the same, tree should predict that value everywhere.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })
    y = pd.Series(np.ones(20) * 5.0, name="y")  # Constant value

    model = TreeRegression(max_depth=3, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    preds = model.predict(X)
    # All predictions should be close to 5.0
    assert all(abs(pred - 5.0) < 0.1 for pred in preds)

def test_max_depth_limitation():
    """
    Test that max_depth parameter actually limits tree depth.
    """
    X = pd.DataFrame({
        "x1": np.arange(10),
        "x2": np.arange(10)
    })
    y = pd.Series(np.arange(10), name="y")

    model = TreeRegression(max_depth=1, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    # With max_depth=1, tree should have at most 2 leaf nodes
    preds = model.predict(X)
    unique_predictions = len(np.unique(preds))
    assert unique_predictions <= 2  # At most 2 different predictions for depth 1

def test_min_samples_split():
    """
    Test that min_samples_split parameter works correctly.
    """
    X = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name="y")

    # With min_samples_split=6, should not split much
    model = TreeRegression(max_depth=3, min_samples_split=6, criterion='mse')
    model.fit(X, y)
    
    preds = model.predict(X)
    # Should have fewer unique predictions due to higher min_samples_split
    unique_predictions = len(np.unique(preds))
    assert unique_predictions <= 4  # Reasonable upper bound

@pytest.mark.parametrize("max_depth, min_samples_split, criterion", [
    (2, 2, 'mse'),
    (3, 3, 'mse'),
    (2, 2, 'mae'),
    (3, 3, 'mae'),
])
def test_hyperparam_variations(max_depth, min_samples_split, criterion):
    """
    Ensure tree runs under different hyperparameters and returns valid results.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })
    y = pd.Series(np.random.rand(20), name="y")

    model = TreeRegression(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    model.fit(X, y)
    score = model.score(X, y)
    
    assert score >= 0.0

@pytest.mark.parametrize("max_depth, min_samples_split, criterion", [
    (0, 2, 'mse'),      # max_depth <= 0
    (2, 0, 'mse'),      # min_samples_split <= 0
    (2, 2, 'invalid'),  # invalid criterion
])
def test_constructor_invalid_params(max_depth, min_samples_split, criterion):
    """
    Constructor should raise ValueError for invalid initialization parameters.
    """
    with pytest.raises(ValueError):
        TreeRegression(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

def test_single_feature():
    """
    Test tree on single-feature data.
    """
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="y")

    model = TreeRegression(max_depth=2, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    # Test prediction on a new point
    test_point = pd.DataFrame({"x": [5.5]})
    pred = model.predict_one(test_point.iloc[0])
    assert isinstance(pred, (int, float))

def test_large_scale_performance():
    """
    Test performance on larger dataset.
    """
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = pd.DataFrame(rng.rand(n_samples, 3), columns=["a", "b", "c"])
    y = pd.Series(1.5*X["a"] - 0.8*X["b"] + 2.2*X["c"] + rng.normal(0, 0.1, n_samples), name="y")
    
    model = TreeRegression(max_depth=5, min_samples_split=10, criterion='mse')
    model.fit(X, y)
    
    # Test on a subset
    X_test = X.iloc[:100]
    y_test = y.iloc[:100]
    preds = model.predict(X_test)
    score = model.score(X_test, y_test)
    
    assert len(preds) == 100
    assert score >= 0.0

def test_edge_case_max_depth_one():
    """
    Test tree with max_depth=1 (stump).
    """
    X = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    y = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], name="y")

    model = TreeRegression(max_depth=1, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    preds = model.predict(X)
    # With max_depth=1, should have at most 2 unique predictions
    unique_predictions = len(np.unique(preds))
    assert unique_predictions <= 2

def test_categorical_like_features():
    """
    Test tree with features that have repeated values (like categorical).
    """
    X = pd.DataFrame({
        "x1": [1, 1, 2, 2, 3, 3, 4, 4],
        "x2": [1, 2, 1, 2, 1, 2, 1, 2]
    })
    y = pd.Series([10, 12, 15, 18, 20, 22, 25, 28], name="y")

    model = TreeRegression(max_depth=3, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    preds = model.predict(X)
    score = model.score(X, y)
    
    assert len(preds) == 8
    assert score >= 0.0

def test_negative_values():
    """
    Test tree with negative feature and target values.
    """
    X = pd.DataFrame({
        "x1": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4],
        "x2": [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8]
    })
    y = pd.Series([-20, -16, -12, -8, -4, 0, 4, 8, 12, 16], name="y")

    model = TreeRegression(max_depth=3, min_samples_split=2, criterion='mse')
    model.fit(X, y)
    
    preds = model.predict(X)
    score = model.score(X, y)
    
    assert len(preds) == 10
    assert score >= 0.0 