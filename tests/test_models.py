"""Tests for the models package."""
from __future__ import annotations

import sys

import polars as pl
import pytest

from models import REGISTRY, PXRModel
from models.baseline import MeanBaseline, MedianBaseline
from models.evaluate import (
    _compute_metrics,
    _evaluate_model,
    _print_results,
    _resolve_models,
    _split_data,
    main,
)
from models.predict import _generate, main as predict_main


@pytest.fixture()
def small_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "SMILES": [f"C{i}" for i in range(10)],
            "pEC50": [float(i) for i in range(10)],
        }
    )


# --- registry ---


def test_registry_contains_baselines() -> None:
    assert "mean_baseline" in REGISTRY
    assert "median_baseline" in REGISTRY


# --- MeanBaseline ---


def test_mean_baseline_fit_predict(small_df: pl.DataFrame) -> None:
    model = MeanBaseline()
    model.fit(small_df, "pEC50")
    preds = model.predict(small_df)
    assert len(preds) == len(small_df)
    assert all(p == pytest.approx(4.5) for p in preds.to_list())


def test_mean_baseline_none_mean() -> None:
    df = pl.DataFrame({"pEC50": pl.Series([], dtype=pl.Float64)})
    model = MeanBaseline()
    model.fit(df, "pEC50")
    assert model._mean == 0.0


# --- MedianBaseline ---


def test_median_baseline_fit_predict(small_df: pl.DataFrame) -> None:
    model = MedianBaseline()
    model.fit(small_df, "pEC50")
    preds = model.predict(small_df)
    assert len(preds) == len(small_df)
    assert all(p == pytest.approx(4.5) for p in preds.to_list())


def test_median_baseline_none_median() -> None:
    df = pl.DataFrame({"pEC50": pl.Series([], dtype=pl.Float64)})
    model = MedianBaseline()
    model.fit(df, "pEC50")
    assert model._median == 0.0


# --- _split_data ---


def test_split_data_sizes(small_df: pl.DataFrame) -> None:
    train, val = _split_data(small_df, 0.3, seed=42)
    assert len(train) + len(val) == len(small_df)
    assert len(val) == 3


def test_split_data_no_overlap(small_df: pl.DataFrame) -> None:
    train, val = _split_data(small_df, 0.2, seed=0)
    assert set(train["SMILES"].to_list()).isdisjoint(set(val["SMILES"].to_list()))


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


def test_evaluate_model(small_df: pl.DataFrame) -> None:
    train, val = _split_data(small_df, 0.3, seed=42)
    metrics = _evaluate_model(MeanBaseline(), train, val, "pEC50")
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


# --- _print_results ---


def test_print_results(capsys: pytest.CaptureFixture[str]) -> None:
    _print_results([("mean_baseline", {"RMSE": 0.5, "MAE": 0.4, "R2": 0.8})])
    out = capsys.readouterr().out
    assert "mean_baseline" in out
    assert "RMSE" in out
    assert "0.5000" in out


# --- main() ---


def test_main_all_models(monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str], small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path)])
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out
    assert "RMSE" in out


def test_main_specific_model(monkeypatch: pytest.MonkeyPatch, tmp_path: object, capsys: pytest.CaptureFixture[str], small_df: pl.DataFrame) -> None:
    csv_path = tmp_path / "train.csv"  # type: ignore[operator]
    small_df.write_csv(csv_path)
    monkeypatch.setattr(sys, "argv", ["evaluate-models", "--train-path", str(csv_path), "--models", "mean_baseline"])
    main()
    out = capsys.readouterr().out
    assert "mean_baseline" in out


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
    train = pl.DataFrame({"Molecule Name": ["A", "B", "C"], "SMILES": ["C", "CC", "CCC"], "pEC50": [5.0, 6.0, 7.0]})
    test = pl.DataFrame({"Molecule Name": ["D", "E"], "SMILES": ["CCCC", "CCCCC"]})
    train_path = tmp_path / "train.csv"  # type: ignore[operator]
    test_path = tmp_path / "test.csv"  # type: ignore[operator]
    train.write_csv(train_path)
    test.write_csv(test_path)
    return train_path, test_path


def test_generate_writes_csv(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "out.csv"  # type: ignore[operator]
    from pathlib import Path

    result = _generate(Path(str(train_path)), Path(str(test_path)), "mean_baseline", "pEC50", Path(str(out)))
    assert out.exists()  # type: ignore[union-attr]
    assert "Molecule Name" in result.columns
    assert "SMILES" in result.columns
    assert "pEC50" in result.columns
    assert len(result) == 2


def test_generate_unknown_model(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Unknown model"):
        _generate(Path(str(train_path)), Path(str(test_path)), "no_such_model", "pEC50", Path(str(tmp_path / "out.csv")))  # type: ignore[operator]


def test_generate_missing_target(tmp_path: object, predict_dfs: tuple[object, object]) -> None:
    train_path, test_path = predict_dfs
    from pathlib import Path

    with pytest.raises(ValueError, match="Target"):
        _generate(Path(str(train_path)), Path(str(test_path)), "mean_baseline", "no_col", Path(str(tmp_path / "out.csv")))  # type: ignore[operator]


# --- predict_main ---


def test_predict_main_explicit_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object], capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path = predict_dfs
    out = tmp_path / "result.csv"  # type: ignore[operator]
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path), "--output", str(out)],
    )
    predict_main()
    assert out.exists()  # type: ignore[union-attr]
    assert "Wrote" in capsys.readouterr().out


def test_predict_main_default_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object, predict_dfs: tuple[object, object], capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path = predict_dfs
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate-results", "--train-path", str(train_path), "--test-path", str(test_path)],
    )
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
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
