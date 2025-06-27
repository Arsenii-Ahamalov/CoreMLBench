# tests/test_logistic_regression.py

import numpy as np
import pandas as pd
import pytest
from src.logistic_regression import LogisticRegression

def test_or_gate_perfect_separation():
    """
    On the OR‐gate (linearly separable with intercept),
    the model should achieve 100% accuracy.
    """
    X = pd.DataFrame({
        "x1": [0, 0, 1, 1],
        "x2": [0, 1, 0, 1]
    })
    y = pd.Series([0, 1, 1, 1], name="y")

    model = LogisticRegression(learning_rate=0.1, n_iters=1000, fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    np.testing.assert_array_equal(preds, y.values)
    assert model.score(X, y) == pytest.approx(1.0)

def test_predict_and_score_types():
    """
    predict returns a numpy array of ints, score returns a float accuracy.
    """
    X = pd.DataFrame({"a": [5,5,5], "b":[-1,-2,-3]})
    y = pd.Series([0, 0, 0], name="y")

    model = LogisticRegression(learning_rate=0.1, n_iters=10, fit_intercept=True)
    model.fit(X, y)

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.dtype == int
    assert preds.shape == (3,)

    acc = model.score(X, y)
    assert isinstance(acc, float)
    assert acc == pytest.approx(1.0)

def test_all_zero_labels():
    """
    If all labels are 0, the classifier should learn to predict 0 everywhere.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(50, 3), columns=["f1","f2","f3"])
    y = pd.Series(np.zeros(50, dtype=int), name="y")

    model = LogisticRegression(learning_rate=0.05, n_iters=200, fit_intercept=True)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.all(preds == 0)
    assert model.score(X, y) == pytest.approx(1.0)

def test_all_one_labels():
    """
    If all labels are 1, the classifier should learn to predict 1 everywhere.
    """
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(30, 2), columns=["u","v"])
    y = pd.Series(np.ones(30, dtype=int), name="y")

    model = LogisticRegression(learning_rate=0.05, n_iters=200, fit_intercept=True)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.all(preds == 1)
    assert model.score(X, y) == pytest.approx(1.0)

def test_linearly_separable_1d():
    """
    On 1D data with threshold at x=0, model should find a positive weight.
    """
    X = pd.DataFrame({"x": np.linspace(-5, 5, 50)})
    y = pd.Series((X["x"] > 0).astype(int), name="y")

    model = LogisticRegression(learning_rate=0.1, n_iters=500, fit_intercept=True)
    model.fit(X, y)

    assert model.weights[0] > 0
    preds = model.predict(X)
    assert model.score(X, y) > 0.95

@pytest.mark.parametrize("learning_rate, n_iters", [
    (0.1, 100),
    (0.01, 500),
])
def test_hyperparam_variations(learning_rate, n_iters):
    """
    Ensure solver runs under different hyperparameters and returns a valid accuracy.
    """
    X = pd.DataFrame({
        "x1":[-1, 0, 1],
        "x2":[ 0, 1, 2]
    })
    y = pd.Series([0, 0, 1], name="y")

    model = LogisticRegression(learning_rate=learning_rate, n_iters=n_iters, fit_intercept=True)
    model.fit(X, y)
    acc = model.score(X, y)
    assert 0.0 <= acc <= 1.0

@pytest.mark.parametrize("lr, n_iters, fit_intercept", [
    (-0.1, 100, True),   # negative learning_rate
    (0.1, -10, True),    # negative n_iters
])
def test_constructor_invalid_params(lr, n_iters, fit_intercept):
    """
    Constructor should raise for invalid initialization parameters.
    """
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=lr, n_iters=n_iters, fit_intercept=fit_intercept)
