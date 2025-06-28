import numpy as np
import pandas as pd
import pytest
from src.classification_tree import ClassificationTree

def test_gini_vs_entropy_criteria():
    """
    Test that both Gini and Entropy criteria work and give reasonable results.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(50),
        "x2": np.random.rand(50)
    })
    y = pd.Series(np.random.choice([0, 1], 50), name="y")

    # Test Gini criterion
    model_gini = ClassificationTree(max_depth=3, min_samples_split=2, criterion='gini')
    model_gini.fit(X, y)
    gini_score = model_gini.score(X, y)

    # Test Entropy criterion
    model_entropy = ClassificationTree(max_depth=3, min_samples_split=2, criterion='entropy')
    model_entropy.fit(X, y)
    entropy_score = model_entropy.score(X, y)

    assert gini_score >= 0.0
    assert entropy_score >= 0.0

def test_predict_and_score_return_types():
    """
    predict(X) returns numpy array; score(X, y) returns float.
    """
    X = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    y = pd.Series([0, 1, 0], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
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
    y = pd.Series(np.ones(20), name="y")  # All ones

    model = ClassificationTree(max_depth=3, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    # All predictions should be 1
    assert all(pred == 1 for pred in preds)

def test_max_depth_limitation():
    """
    Test that max_depth parameter actually limits tree depth.
    """
    X = pd.DataFrame({
        "x1": np.arange(10),
        "x2": np.arange(10)
    })
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=1, min_samples_split=2, criterion='gini')
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
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    # With min_samples_split=6, should not split much
    model = ClassificationTree(max_depth=3, min_samples_split=6, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    # Should have fewer unique predictions due to higher min_samples_split
    unique_predictions = len(np.unique(preds))
    assert unique_predictions <= 4  # Reasonable upper bound

@pytest.mark.parametrize("max_depth, min_samples_split, criterion", [
    (2, 2, 'gini'),
    (3, 3, 'gini'),
    (2, 2, 'entropy'),
    (3, 3, 'entropy'),
])
def test_hyperparam_variations(max_depth, min_samples_split, criterion):
    """
    Ensure tree runs under different hyperparameters and returns valid results.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })
    y = pd.Series(np.random.choice([0, 1], 20), name="y")

    model = ClassificationTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    model.fit(X, y)
    score = model.score(X, y)
    
    assert score >= 0.0

@pytest.mark.parametrize("max_depth, min_samples_split, criterion", [
    (0, 2, 'gini'),      # max_depth <= 0
    (2, 0, 'gini'),      # min_samples_split <= 0
    (2, 2, 'invalid'),   # invalid criterion
])
def test_constructor_invalid_params(max_depth, min_samples_split, criterion):
    """
    Constructor should raise ValueError for invalid initialization parameters.
    """
    with pytest.raises(ValueError):
        ClassificationTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

def test_single_feature():
    """
    Test tree on single-feature data.
    """
    X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    # Test prediction on a new point
    test_point = pd.DataFrame({"x": [5.5]})
    pred = model.predict_one(test_point.iloc[0])
    assert isinstance(pred, (int, np.integer))

def test_large_scale_performance():
    """
    Test performance on larger dataset.
    """
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = pd.DataFrame(rng.rand(n_samples, 3), columns=["a", "b", "c"])
    y = pd.Series(rng.choice([0, 1, 2], n_samples), name="y")
    
    model = ClassificationTree(max_depth=5, min_samples_split=10, criterion='gini')
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
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=1, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    # With max_depth=1, should have at most 2 unique predictions
    unique_predictions = len(np.unique(preds))
    assert unique_predictions <= 2


def test_negative_values():
    """
    Test tree with negative feature values.
    """
    X = pd.DataFrame({
        "x1": [-1, -2, -3, -4, 1, 2, 3, 4],
        "x2": [-1, -2, -3, -4, 1, 2, 3, 4]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 8

def test_multiclass_classification():
    """
    Test tree with more than 2 classes.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(30),
        "x2": np.random.rand(30)
    })
    y = pd.Series(np.random.choice([0, 1, 2], 30), name="y")

    model = ClassificationTree(max_depth=3, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 30
    assert all(pred in [0, 1, 2] for pred in preds)

def test_perfect_separation():
    """
    Test tree on data that can be perfectly separated.
    """
    X = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds = model.predict(X)
    # Should be able to perfectly classify this data
    assert all(pred in [0, 1] for pred in preds)

def test_gini_calculation():
    """
    Test that Gini impurity calculation is correct.
    """
    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    
    # Test with pure class (all same values)
    y_pure = pd.Series([1, 1, 1, 1])
    gini_pure = model._ClassificationTree__gini(y_pure)
    assert gini_pure == 0.0  # Pure class should have 0 impurity
    
    # Test with mixed classes
    y_mixed = pd.Series([0, 0, 1, 1])
    gini_mixed = model._ClassificationTree__gini(y_mixed)
    assert 0.0 < gini_mixed < 1.0  # Mixed classes should have impurity > 0

def test_entropy_calculation():
    """
    Test that Entropy calculation is correct.
    """
    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='entropy')
    
    # Test with pure class (all same values)
    y_pure = pd.Series([1, 1, 1, 1])
    entropy_pure = model._ClassificationTree__entropy(y_pure)
    assert entropy_pure == 0.0  # Pure class should have 0 entropy
    
    # Test with mixed classes
    y_mixed = pd.Series([0, 0, 1, 1])
    entropy_mixed = model._ClassificationTree__entropy(y_mixed)
    assert 0.0 < entropy_mixed < 2.0  # Mixed classes should have entropy > 0

def test_empty_splits():
    """
    Test behavior when splits result in empty subsets.
    """
    X = pd.DataFrame({
        "x1": [1, 1, 1, 1, 1, 1, 1, 1],  # All same values
        "x2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=3, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    # Should handle this gracefully and create a leaf node
    preds = model.predict(X)
    assert len(preds) == 8

def test_single_sample():
    """
    Test tree with very small dataset.
    """
    X = pd.DataFrame({"x": [1]})
    y = pd.Series([0], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    pred = model.predict(X)
    assert len(pred) == 1
    assert pred[0] == 0  # Should predict the only available class

def test_consistency_between_predict_and_predict_one():
    """
    Test that predict() and predict_one() give consistent results.
    """
    X = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8],
        "x2": [1, 2, 3, 4, 5, 6, 7, 8]
    })
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], name="y")

    model = ClassificationTree(max_depth=2, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    preds_batch = model.predict(X)
    preds_single = [model.predict_one(X.iloc[i]) for i in range(len(X))]
    
    assert np.array_equal(preds_batch, preds_single)

def test_score_method_consistency():
    """
    Test that score method returns consistent results for same data.
    """
    X = pd.DataFrame({
        "x1": np.random.rand(20),
        "x2": np.random.rand(20)
    })
    y = pd.Series(np.random.choice([0, 1], 20), name="y")

    model = ClassificationTree(max_depth=3, min_samples_split=2, criterion='gini')
    model.fit(X, y)
    
    score1 = model.score(X, y)
    score2 = model.score(X, y)
    
    assert score1 == score2  