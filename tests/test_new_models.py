"""Tests for KNN, DecisionTree, LinearRegression, and XGBoost models."""
from __future__ import annotations

import numpy as np
import pytest

from models.decision_tree import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_RANDOM_STATE,
    DecisionTree,
)
from models.knn import DEFAULT_METRIC, DEFAULT_N_NEIGHBORS, DEFAULT_WEIGHTS, KNN
from models.linear_regression import DEFAULT_ALPHA, DEFAULT_FIT_INTERCEPT, DEFAULT_MAX_ITER, DEFAULT_N_COMPONENTS, LinearRegression
from models.xgboost_model import (
    DEFAULT_COLSAMPLE_BYTREE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_SUBSAMPLE,
    XGBoost,
)


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 300))
    y = X[:, 0] * 2.0 + rng.standard_normal(500) * 0.1
    X_test = rng.standard_normal((50, 300))
    return X, y, X_test


# --- KNN ---


def test_knn_default_params() -> None:
    model = KNN()
    assert model._model.n_neighbors == DEFAULT_N_NEIGHBORS
    assert model._model.weights == DEFAULT_WEIGHTS
    assert model._model.metric == DEFAULT_METRIC


def test_knn_custom_params() -> None:
    model = KNN(n_neighbors=3, weights="distance", metric="manhattan")
    assert model._model.n_neighbors == 3
    assert model._model.weights == "distance"
    assert model._model.metric == "manhattan"


def test_knn_fit_predict(regression_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y, X_test = regression_data
    model = KNN()
    model.fit(X, y)
    preds = model.predict(X_test)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


# --- DecisionTree ---


def test_decision_tree_default_params() -> None:
    model = DecisionTree()
    assert model._model.max_depth == DEFAULT_MAX_DEPTH
    assert model._model.min_samples_split == DEFAULT_MIN_SAMPLES_SPLIT
    assert model._model.min_samples_leaf == DEFAULT_MIN_SAMPLES_LEAF
    assert model._model.random_state == DEFAULT_RANDOM_STATE


def test_decision_tree_custom_params() -> None:
    model = DecisionTree(max_depth=3, min_samples_split=4, min_samples_leaf=2, random_state=7)
    assert model._model.max_depth == 3
    assert model._model.min_samples_split == 4
    assert model._model.min_samples_leaf == 2
    assert model._model.random_state == 7


def test_decision_tree_none_max_depth() -> None:
    model = DecisionTree(max_depth=None)
    assert model._model.max_depth is None


def test_decision_tree_fit_predict(regression_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y, X_test = regression_data
    model = DecisionTree()
    model.fit(X, y)
    preds = model.predict(X_test)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


# --- LinearRegression ---


def test_linear_regression_default_params() -> None:
    model = LinearRegression()
    ridge = model._model.named_steps["ridge"]
    assert ridge.alpha == pytest.approx(DEFAULT_ALPHA)
    assert ridge.fit_intercept == DEFAULT_FIT_INTERCEPT
    assert ridge.max_iter == DEFAULT_MAX_ITER
    assert model._model.named_steps["pca"].n_components == DEFAULT_N_COMPONENTS
    assert "variance_threshold" in model._model.named_steps
    assert "scaler" in model._model.named_steps


def test_linear_regression_custom_params() -> None:
    model = LinearRegression(alpha=1.0, fit_intercept=False, max_iter=500, n_components=50)
    ridge = model._model.named_steps["ridge"]
    assert ridge.alpha == pytest.approx(1.0 / 50)
    assert ridge.fit_intercept is False
    assert ridge.max_iter == 500
    assert model._model.named_steps["pca"].n_components == 50


def test_linear_regression_fit_predict(regression_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y, X_test = regression_data
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X_test)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


# --- XGBoost ---


def test_xgboost_default_params() -> None:
    model = XGBoost()
    assert model._model.n_estimators == DEFAULT_N_ESTIMATORS
    assert model._model.learning_rate == pytest.approx(DEFAULT_LEARNING_RATE)
    assert model._model.subsample == pytest.approx(DEFAULT_SUBSAMPLE)
    assert model._model.colsample_bytree == pytest.approx(DEFAULT_COLSAMPLE_BYTREE)


def test_xgboost_custom_params() -> None:
    model = XGBoost(n_estimators=50, max_depth=3, learning_rate=0.05)
    assert model._model.n_estimators == 50
    assert model._model.max_depth == 3
    assert model._model.learning_rate == pytest.approx(0.05)


def test_xgboost_fit_predict(regression_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    X, y, X_test = regression_data
    model = XGBoost()
    model.fit(X, y)
    preds = model.predict(X_test)
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()


# --- registry ---


def test_all_new_models_in_registry() -> None:
    from models import REGISTRY
    for name in ("knn", "decision_tree", "linear_regression", "xgboost"):
        assert name in REGISTRY
