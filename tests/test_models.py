"""Tests for the models package."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest

from pathlib import Path

from models import REGISTRY, META_REGISTRY
from models.baseline import MeanBaseline, MedianBaseline
from models.delta_model import DeltaModel
from models.knn import KNN
from models.stacked_ensemble import StackedEnsemble
from models.xgboost_model import XGBoost
from models.evaluate import (
    _compute_metrics,
    _evaluate_model,
    _print_results,
    _resolve_input,
    _resolve_models,
    main,
)
from models.predict import _generate, main as predict_main


@pytest.fixture()
def small_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "SMILES": ["c1ccccc1", "CCO", "CC(=O)O", "CC", "CCC", "CCCC", "c1ccncc1", "CCN", "CC(C)O", "CCCCO"],
            "pEC50": [float(i) for i in range(10)],
        }
    )


@pytest.fixture()
def data_dir(tmp_path: object, small_df: pl.DataFrame) -> object:
    """Write train.csv and test_unblinded.csv so tests can use the default unblinded split."""
    smiles = small_df["SMILES"].to_list()
    train_path = tmp_path / "train.csv"  # type: ignore[operator]
    unblinded_path = tmp_path / "test_unblinded.csv"  # type: ignore[operator]
    pl.DataFrame({"SMILES": smiles[:7], "pEC50": [float(i) for i in range(7)]}).write_csv(train_path)
    pl.DataFrame(
        {
            "SMILES": smiles[7:],
            "pEC50": [float(i) for i in range(3)],
            "OCNT Batch": [f"OCNT-{i:04d}-01" for i in range(3)],
        }
    ).write_csv(unblinded_path)
    return tmp_path


@pytest.fixture()
def small_xy() -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((10, 4), dtype=np.float64)
    y = np.arange(10, dtype=np.float64)
    return X, y


# --- registry ---


def test_registry_contains_baselines() -> None:
    assert "mean_baseline" in REGISTRY
    assert "median_baseline" in REGISTRY


# --- MeanBaseline ---


def test_mean_baseline_fit_predict(small_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_xy
    model = MeanBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p == pytest.approx(4.5) for p in preds)


def test_mean_baseline_empty_y() -> None:
    model = MeanBaseline()
    model.fit(np.zeros((0, 4)), np.array([]))
    assert model._mean == 0.0


# --- MedianBaseline ---


def test_median_baseline_fit_predict(small_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_xy
    model = MedianBaseline()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(p == pytest.approx(4.5) for p in preds)


def test_median_baseline_empty_y() -> None:
    model = MedianBaseline()
    model.fit(np.zeros((0, 4)), np.array([]))
    assert model._median == 0.0


# --- _compute_metrics ---


def test_compute_metrics_perfect() -> None:
    a = pl.Series([1.0, 2.0, 3.0])
    metrics = _compute_metrics(a, a)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


def test_compute_metrics_constant_actual() -> None:
    a = pl.Series([5.0, 5.0, 5.0])
    p = pl.Series([4.0, 5.0, 6.0])
    metrics = _compute_metrics(a, p)
    assert metrics["R2"] == pytest.approx(0.0)


# --- _evaluate_model ---


def test_evaluate_model(small_xy: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_xy
    X_train, y_train = X[:7], y[:7]
    X_val, y_val = X[7:], y[7:]
    metrics = _evaluate_model(MeanBaseline(), X_train, y_train, X_val, y_val)
    assert set(metrics.keys()) == {"RMSE", "MAE", "R2"}
    assert metrics["RMSE"] >= 0.0


# --- _resolve_models ---


def test_resolve_all() -> None:
    classes = _resolve_models(["all"])
    assert {c.name for c in classes} == set(REGISTRY.keys())


def test_resolve_specific() -> None:
    classes = _resolve_models(["mean_baseline"])
    assert len(classes) == 1
    assert classes[0].name == "mean_baseline"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown models"):
        _resolve_models(["not_a_model"])


# --- _resolve_input ---


def test_resolve_input_valid() -> None:
    _resolve_input("morgan")  # no exception


def test_resolve_input_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown input"):
        _resolve_input("not_an_input")


# --- _print_results ---


def test_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    _print_results([("mean_baseline", {"RMSE": 0.5, "MAE": 0.4, "R2": 0.8})])
    out = capsys.readouterr().out
    assert "mean_baseline" in out
    assert "RMSE" in out
    assert "0.5000" in out


# --- evaluate main() ---


def test_main_all_models(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-models",
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            str(data_dir) + "/cache",
            "--models",
            "mean_baseline",
            "median_baseline",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out
    assert "RMSE" in out


def test_main_specific_model(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-models",
            "--data-dir",
            str(data_dir),
            "--models",
            "mean_baseline",
            "--cache-dir",
            str(data_dir) + "/cache",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out


def test_main_custom_input(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: object,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate-models",
            "--data-dir",
            str(data_dir),
            "--models",
            "mean_baseline",
            "--input",
            "rdkit",
            "morgan",
            "--cache-dir",
            str(data_dir) + "/cache",
        ],
    )
    main()
    assert "mean_baseline" in capsys.readouterr().out


def test_main_bad_val_split(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setattr(
        sys, "argv", ["evaluate-models", "--scaffold-split", "--data-dir", str(tmp_path), "--val-split", "0.0"]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_unknown_model(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--data-dir", str(tmp_path), "--models", "no_such_model"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_unknown_input(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--data-dir", str(tmp_path), "--input", "no_such_input"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_missing_target(
    monkeypatch: pytest.MonkeyPatch,
    data_dir: object,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate-models", "--data-dir", str(data_dir), "--models", "mean_baseline", "--target", "no_column"],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- predict._generate ---


@pytest.fixture()
def predict_dfs(tmp_path: object) -> tuple[object, object]:
    train = pl.DataFrame(
        {"Molecule Name": ["A", "B", "C"], "SMILES": ["c1ccccc1", "CCO", "CCC"], "pEC50": [5.0, 6.0, 7.0]}
    )
    test = pl.DataFrame({"Molecule Name": ["D", "E"], "SMILES": ["CCCC", "c1ccncc1"]})
    train_path = tmp_path / "train.csv"  # type: ignore[operator]
    test_path = tmp_path / "test.csv"  # type: ignore[operator]
    train.write_csv(train_path)
    test.write_csv(test_path)
    return train_path, test_path


def test_generate_writes_csv(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "out.csv"  # type: ignore[operator]
    from pathlib import Path

    cache = tmp_path / "cache"  # type: ignore[operator]
    result = _generate(
        Path(str(train_path)),
        Path(str(test_path)),
        "mean_baseline",
        ["morgan"],
        "pEC50",
        Path(str(out)),
        Path(str(cache)),
    )
    assert out.exists()  # type: ignore[union-attr]
    assert "Molecule Name" in result.columns
    assert "SMILES" in result.columns
    assert "pEC50" in result.columns
    assert len(result) == 2


def test_generate_unknown_model(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Unknown model"):
        _generate(
            Path(str(train_path)),
            Path(str(test_path)),
            "no_such_model",
            ["morgan"],
            "pEC50",
            Path(str(tmp_path / "out.csv")),
            Path(str(tmp_path / "cache")),  # type: ignore[operator]
        )


def test_generate_unknown_input(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Unknown input"):
        _generate(
            Path(str(train_path)),
            Path(str(test_path)),
            "mean_baseline",
            ["no_input"],
            "pEC50",
            Path(str(tmp_path / "out.csv")),
            Path(str(tmp_path / "cache")),  # type: ignore[operator]
        )


def test_generate_missing_target(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Target"):
        _generate(
            Path(str(train_path)),
            Path(str(test_path)),
            "mean_baseline",
            ["morgan"],
            "no_col",
            Path(str(tmp_path / "out.csv")),
            Path(str(tmp_path / "cache")),  # type: ignore[operator]
        )


# --- predict_main ---


def test_predict_main_explicit_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
    predict_dfs: tuple[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "result.csv"  # type: ignore[operator]
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate-results",
            "--train-path",
            str(train_path),
            "--test-path",
            str(test_path),
            "--output",
            str(out),
            "--cache-dir",
            str(cache),
        ],
    )
    predict_main()
    assert out.exists()  # type: ignore[union-attr]
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_default_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
    predict_dfs: tuple[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_path, test_path = predict_dfs
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--cache-dir", str(cache)],
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    predict_main()
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_custom_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
    predict_dfs: tuple[object, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "result.csv"  # type: ignore[operator]
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate-results",
            "--train-path",
            str(train_path),
            "--test-path",
            str(test_path),
            "--input",
            "rdkit",
            "--output",
            str(out),
            "--cache-dir",
            str(cache),
        ],
    )
    predict_main()
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_unknown_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]
) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--model", "no_model"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1


def test_predict_main_unknown_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]
) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--input", "no_input"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1


def test_predict_main_missing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]
) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--target", "no_col"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1


# --- DeltaModel ---


def _clustered_xy() -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated direction clusters with distinct pEC50 levels.

    Cluster A (rows 0-5) points along [1,1,1,0], pEC50 ~5; cluster B (rows 6-9)
    along [0,0,1,1], pEC50 ~2. Small noise keeps every column non-constant.
    """
    rng = np.random.default_rng(0)
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    rows = [a * c + rng.normal(0, 0.01, 4) for c in (1, 2, 3, 4, 5, 6)]
    rows += [b * c + rng.normal(0, 0.01, 4) for c in (1, 2, 3, 4)]
    X = np.asarray(rows, dtype=np.float64)
    y = np.array([5.0] * 6 + [2.0] * 4, dtype=np.float64)
    return X, y


def _small_delta() -> DeltaModel:
    return DeltaModel(embedding_dim=16, epochs=3, n_pairs_per_epoch=200)


def test_delta_registered() -> None:
    assert "delta" in REGISTRY
    assert REGISTRY["delta"] is DeltaModel


def test_delta_fit_predict_shape() -> None:
    X, y = _clustered_xy()
    model = _small_delta()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_delta_abstains_when_no_neighbors() -> None:
    X, y = _clustered_xy()
    model = _small_delta()
    model.fit(X, y)
    # Direction orthogonal to both clusters -> cosine ~0 < SIM_CUTOFF for every train row.
    X_test = np.array([[1.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0]])
    preds = model.predict(X_test)
    assert np.all(preds == 0.0)


def test_delta_recovers_neighbor_activity() -> None:
    X, y = _clustered_xy()
    model = _small_delta()
    model.fit(X, y)
    # A point in cluster A's direction is covered by its 6 neighbors (all pEC50 ~5).
    preds = model.predict(np.array([[1.0, 1.0, 1.0, 0.0]]))
    assert preds[0] != 0.0  # covered, not abstained
    assert preds[0] == pytest.approx(5.0, abs=1.0)


def test_find_cliff_pairs_detects_obvious_cliff() -> None:
    # Rows 0,1 share a direction but differ in pEC50 by 2.0 -> one cliff pair.
    X = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    y = np.array([6.0, 4.0, 5.0])
    pairs = DeltaModel()._find_cliff_pairs(X, y)
    assert pairs.shape == (1, 2)
    assert set(pairs[0].tolist()) == {0, 1}


# --- StackedEnsemble ---

_STACK_SMILES = [
    "c1ccccc1",
    "CCO",
    "CC(=O)O",
    "CC",
    "CCC",
    "CCCC",
    "c1ccncc1",
    "CCN",
    "CC(C)O",
    "CCCCO",
    "CCCC(=O)O",
    "c1ccc(O)cc1",
    "CCOCC",
    "CCCCCC",
]


def _stack_df() -> pl.DataFrame:
    n = len(_STACK_SMILES)
    return pl.DataFrame(
        {
            "Molecule Name": [f"M{i}" for i in range(n)],
            "SMILES": _STACK_SMILES,
            "pEC50": [4.0 + (i % 5) * 0.5 for i in range(n)],
        }
    )


def _cheap_roster():
    """A roster using only offline-computable reps/models (no pretrained embeddings)."""
    return [
        ("xgb", XGBoost, ["morgan"]),
        ("knn", KNN, ["morgan"]),
        ("delta", lambda: DeltaModel(epochs=2, n_pairs_per_epoch=100), ["rdkit"]),
    ]


def test_stacked_ensemble_registered() -> None:
    assert "stacked_ensemble" in META_REGISTRY
    assert META_REGISTRY["stacked_ensemble"] is StackedEnsemble


def test_stacked_ensemble_fit_predict(tmp_path: object) -> None:
    cache = Path(str(tmp_path)) / "cache"  # type: ignore[operator]
    train = _stack_df()
    test = _stack_df().head(4)
    model = StackedEnsemble(roster=_cheap_roster(), n_splits=2)
    model.fit(train, cache)
    preds = model.predict(test, cache)
    assert preds.shape == (len(test),)
    assert np.all(np.isfinite(preds))
