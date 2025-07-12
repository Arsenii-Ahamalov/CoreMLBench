import numpy as np
import pandas as pd
import pytest
from src.linear_regression import LinearRegression

@pytest.mark.parametrize("solver", ["normal", "gd"])
def test_perfect_linearity_pandas(solver):
    """
    On noiseless data y = 3*x (zero intercept),
    both solvers should recover weight≈3 and bias≈0.
    """
    X = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series(3 * X["x"], name="y")

    model = LinearRegression(solver=solver, learning_rate=0.01, n_iters=1000)
    model.fit(X, y)

    assert model.weights is not None
    w = model.weights[0]
    b = model.bias

    assert pytest.approx(w, rel=1e-2) == 3
    assert pytest.approx(b, abs=1e-2) == 0

def test_fit_with_noise_pandas():
    """
    On data y = 5*x + small noise, GD solver should recover
    weight≈5 and bias≈0 (rather than testing raw MSE).
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(100, 1), columns=pd.Index(["f"]))
    y_true = 5 * X["f"]
    y = pd.Series(y_true + 0.1 * rng.randn(100), name="y")

    model = LinearRegression(solver="gd", learning_rate=0.01, n_iters=5000)
    model.fit(X, y)

    assert model.weights is not None
    # Check recovered parameters rather than raw MSE
    w = model.weights[0]
    b = model.bias

    # weight close to 5, bias close to 0
    assert pytest.approx(w, rel=1e-1) == 5   # allow 10% tolerance for noisy data
    assert pytest.approx(b, abs=0.1) == 0

def test_normal_vs_gd_agreement_pandas():
    """
    Compare normal-equation vs. GD on multivariate data with intercept.
    Both weight vectors and biases must agree within tolerance.
    """
    rng = np.random.RandomState(1)
    data = rng.randn(50, 3)
    X = pd.DataFrame(data, columns=pd.Index(["a", "b", "c"]))
    true_w = np.array([1.5, -2.0, 0.7])
    true_b = 0.5
    y = pd.Series(X.values.dot(true_w) + true_b, name="y")

    m1 = LinearRegression(solver="normal")
    m1.fit(X, y)
    m2 = LinearRegression(solver="gd", learning_rate=0.01, n_iters=5000)
    m2.fit(X, y)

    assert m1.weights is not None and m2.weights is not None
    assert np.allclose(np.asarray(m1.weights), np.asarray(m2.weights), atol=1e-2)
    assert pytest.approx(m1.bias, rel=1e-2) == m2.bias

def test_predict_and_score_return_types():
    """
    predict(X) returns a numpy array; score(X, y) returns a float R^2.
    """
    X = pd.DataFrame({"x1": [0,1,2], "x2": [3,4,5]})
    # y = 1*x1 + 2*x2 + 0
    y = pd.Series(1*X["x1"] + 2*X["x2"], name="y")

    model = LinearRegression(solver="normal")
    model.fit(X, y)

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    np.testing.assert_allclose(np.asarray(preds), np.asarray(y.values))

    r2 = model.score(X, y)
    assert isinstance(r2, float)
    assert r2 == pytest.approx(1.0)

def test_singular_matrix_pandas():
    """
    If two columns are identical, normal solver must not error,
    and predictions (with recovered bias) still match y.
    """
    base = np.arange(5)
    X = pd.DataFrame({"c1": base, "c2": base})
    y = pd.Series(2 * X["c1"] + 1.0, name="y")  # include bias=1

    model = LinearRegression(solver="normal")
    model.fit(X, y)

    # predictions using both weight and bias
    preds = model.predict(X)
    np.testing.assert_allclose(np.asarray(preds), np.asarray(y.values), atol=1e-6)

@pytest.mark.parametrize("solver, lr, n_iters", [
    ("normal", 0.01, 100),   # normal ignores lr/n_iters
    ("gd",     0.01, 1000),
])
def test_constructor_and_trivial(solver, lr, n_iters):
    """
    Quick sanity: trivial data y=0 gives zero weights and bias.
    """
    X = pd.DataFrame(np.zeros((10,2)), columns=pd.Index(["a","b"]))
    y = pd.Series(np.zeros(10), name="y")

    model = LinearRegression(solver=solver, learning_rate=lr, n_iters=n_iters)
    model.fit(X, y)

    assert model.weights is not None
    assert np.allclose(np.asarray(model.weights), 0)
    assert pytest.approx(model.bias, abs=1e-6) == 0

@pytest.mark.parametrize("solver, n_iters, lr", [
    # invalid init parameters still raises
    ("bad", 100, 0.1),
    ("gd", -1, 0.1),
    ("normal", 100, -0.1),
])
def test_constructor_invalid_params(solver, n_iters, lr):
    with pytest.raises(ValueError):
        LinearRegression(solver=solver, learning_rate=lr, n_iters=n_iters)

def test_large_scale_performance():
    """
    Test performance on large dataset (10k samples)
    """
    rng = np.random.RandomState(42)
    n_samples = 10000
    X = pd.DataFrame(rng.rand(n_samples, 3), columns=pd.Index(["a", "b", "c"]))
    y = pd.Series(1.5*X["a"] - 0.8*X["b"] + 2.2*X["c"] + 0.5, name="y")
    
    model = LinearRegression(solver="normal")
    model.fit(X, y)
    
    preds = model.predict(X)
    r2 = model.score(X, y)
    assert r2 > 0.99999

def test_missing_values_handling():
    """
    Test proper error handling with missing values
    """
    X = pd.DataFrame({"x": [1, 2, np.nan, 4]})
    y = pd.Series([2, 4, 6, 8], name="y")
    
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_categorical_features_error():
    """
    Test proper error with categorical features
    """
    X = pd.DataFrame({"category": ["A", "B", "C", "A"]})
    y = pd.Series([10, 20, 30, 15], name="y")
    
    model = LinearRegression()
    with pytest.raises(TypeError):
        model.fit(X, y)

def test_negative_weights_recovery():
    """
    Test recovery of negative weights
    """
    X = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series(-3 * X["x"] + 5, name="y")  # y = -3x + 5
    
    model = LinearRegression(solver="gd", learning_rate=0.01, n_iters=2000)
    model.fit(X, y)
    
    assert model.weights is not None
    assert pytest.approx(model.weights[0], abs=0.1) == -3
    assert pytest.approx(model.bias, abs=0.1) == 5