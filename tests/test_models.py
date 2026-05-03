"""Tests for the models package."""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest

from models import REGISTRY, PXRModel
from models.baseline import MeanBaseline, MedianBaseline
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str], small_df: pl.DataFrame
) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(
        sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--cache-dir", str(tmp_path / "cache")]
    )
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out
    assert "RMSE" in out


def test_main_specific_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str], small_df: pl.DataFrame
) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate-models", "--train-path", str(csv_path), "--models", "mean_baseline", "--cache-dir", str(tmp_path / "cache")],
    )
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out


def test_main_custom_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str], small_df: pl.DataFrame
) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate-models", "--train-path", str(csv_path), "--input", "rdkit+morgan", "--cache-dir", str(tmp_path / "cache")],
    )
    main()
    assert "mean_baseline" in capsys.readouterr().out


def test_main_bad_val_split(monkeypatch: pytest.MonkeyPatch, tmp_path: object, small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--val-split", "0.0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_unknown_model(monkeypatch: pytest.MonkeyPatch, tmp_path: object, small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--models", "no_such_model"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_unknown_input(monkeypatch: pytest.MonkeyPatch, tmp_path: object, small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--input", "no_such_input"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_main_missing_target(monkeypatch: pytest.MonkeyPatch, tmp_path: object, small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--target", "no_column"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


# --- predict._generate ---


@pytest.fixture()
def predict_dfs(tmp_path: object) -> tuple[object, object]:
    train = pl.DataFrame({"Molecule Name": ["A", "B", "C"], "SMILES": ["c1ccccc1", "CCO", "CCC"], "pEC50": [5.0, 6.0, 7.0]})
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
    result = _generate(Path(str(train_path)), Path(str(test_path)), "mean_baseline", "morgan", "pEC50", Path(str(out)), Path(str(cache)))
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
            Path(str(train_path)), Path(str(test_path)), "no_such_model", "morgan", "pEC50",
            Path(str(tmp_path / "out.csv")), Path(str(tmp_path / "cache"))  # type: ignore[operator]
        )


def test_generate_unknown_input(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Unknown input"):
        _generate(
            Path(str(train_path)), Path(str(test_path)), "mean_baseline", "no_input", "pEC50",
            Path(str(tmp_path / "out.csv")), Path(str(tmp_path / "cache"))  # type: ignore[operator]
        )


def test_generate_missing_target(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Target"):
        _generate(
            Path(str(train_path)), Path(str(test_path)), "mean_baseline", "morgan", "no_col",
            Path(str(tmp_path / "out.csv")), Path(str(tmp_path / "cache"))  # type: ignore[operator]
        )


# --- predict_main ---


def test_predict_main_explicit_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object], capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "result.csv"  # type: ignore[operator]
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path),
         "--output", str(out), "--cache-dir", str(cache)],
    )
    predict_main()
    assert out.exists()  # type: ignore[union-attr]
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_default_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object], capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path = predict_dfs
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path),
         "--cache-dir", str(cache)],
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    predict_main()
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_custom_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object], capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "result.csv"  # type: ignore[operator]
    cache = tmp_path / "cache"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate-results",
            "--train-path", str(train_path),
            "--test-path", str(test_path),
            "--input", "rdkit",
            "--output", str(out),
            "--cache-dir", str(cache),
        ],
    )
    predict_main()
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_unknown_model(monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--model", "no_model"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1


def test_predict_main_unknown_input(monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--input", "no_input"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1


def test_predict_main_missing_target(monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--target", "no_col"],
    )
    with pytest.raises(SystemExit) as exc:
        predict_main()
    assert exc.value.code == 1
