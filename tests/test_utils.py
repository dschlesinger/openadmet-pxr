"""Tests for models.utils."""
from __future__ import annotations

import numpy as np
import pytest

from models.utils import drop_nan_rows


def test_no_nans_unchanged() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([0.0, 1.0])
    X_out, y_out = drop_nan_rows(X, y)
    assert X_out.shape == (2, 2)
    assert y_out is not None and len(y_out) == 2


def test_drops_nan_rows_with_y() -> None:
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 1.0]])
    y = np.array([0.0, 1.0, 2.0])
    X_out, y_out = drop_nan_rows(X, y)
    assert X_out.shape == (1, 2)
    assert y_out is not None and y_out.tolist() == [1.0]


def test_drops_inf_rows() -> None:
    X = np.array([[1.0, np.inf], [3.0, 4.0], [-np.inf, 1.0]])
    y = np.array([0.0, 1.0, 2.0])
    X_out, y_out = drop_nan_rows(X, y)
    assert X_out.shape == (1, 2)
    assert y_out is not None and y_out.tolist() == [1.0]


def test_drops_nan_rows_without_y() -> None:
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    X_out, y_out = drop_nan_rows(X)
    assert X_out.shape == (1, 2)
    assert y_out is None


def test_prints_dropped_count(capsys: pytest.CaptureFixture[str]) -> None:
    X = np.array([[np.nan, 1.0], [2.0, 3.0]])
    drop_nan_rows(X, label="train")
    out = capsys.readouterr().out
    assert "Dropped 1 of 2" in out
    assert "train" in out


def test_prints_label_in_message(capsys: pytest.CaptureFixture[str]) -> None:
    X = np.array([[np.inf, 1.0], [2.0, 3.0]])
    drop_nan_rows(X, label="val")
    out = capsys.readouterr().out
    assert "val" in out


def test_no_drop_no_message(capsys: pytest.CaptureFixture[str]) -> None:
    X = np.array([[1.0, 2.0]])
    drop_nan_rows(X, np.array([1.0]))
    assert capsys.readouterr().out == ""
