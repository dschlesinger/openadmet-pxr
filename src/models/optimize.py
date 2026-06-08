"""Optuna hyperparameter search spaces for PXR challenge models."""

from typing import Any, Callable, Dict

import optuna

SearchSpace = Callable[[optuna.Trial], Dict[str, Any]]

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "xgboost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        "random_state": 42,
    },
    "knn": lambda trial: {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
    },
    "decision_tree": lambda trial: {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    },
    "mlp": lambda trial: {
        "hidden_dims": tuple(
            trial.suggest_int(f"layer_{i}", 32, 512, log=True)
            for i in range(trial.suggest_int("n_layers", 1, 4))
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "max_epochs": trial.suggest_int("max_epochs", 100, 500, step=100),
        "patience": trial.suggest_int("patience", 10, 50, step=10),
    },
    "linear_regression": lambda trial: {
        "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "n_components": trial.suggest_int("n_components", 10, 500, log=True),
    },
    "symbolic_regression": lambda trial: {
        "niterations": trial.suggest_int("niterations", 10, 100),
        "populations": trial.suggest_int("populations", 5, 50),
    },
    "tabicl": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 1, 32),
        "pca_components": trial.suggest_int("pca_components", 10, 300, log=True),
    },
    "tabpfn": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 1, 32),
        "pca_components": trial.suggest_int("pca_components", 10, 300, log=True),
    },
    "delta": lambda trial: {
        "embedding_dim": trial.suggest_int("embedding_dim", 32, 512, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "epochs": trial.suggest_int("epochs", 20, 150, step=10),
        "n_pairs_per_epoch": trial.suggest_int("n_pairs_per_epoch", 5_000, 50_000, log=True),
    },
}
