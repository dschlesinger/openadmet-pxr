"""Stacked ensemble meta-model for the PXR challenge.

A two-level stack (discoverybytes rank 11 / mp-alex rank 64 blueprint):

  level-0  a diverse roster of (model x representation) base learners
  level-1  an ElasticNetCV meta-learner trained on out-of-fold (OOF) predictions
           plus a few uncertainty meta-features

The roster spans orthogonal signal families (2D-pretrained, 3D, tabular
foundation, tabular descriptors/QM, local, relational) chosen from a
score-table curation; os_reports stresses *representation* diversity over seed
replication. OOF predictions come from honest Butina grouped K-fold (no similar
molecules span a train/val boundary). The meta-learner is fit on OOF only — no
in-sample calibration, no leakage.

Implementation notes:
  - Featurize-once + row-slice: every representation is featurized on the full
    training set a single time, then CV folds index rows. This avoids
    re-featurizing subsets (which would miss the cache and, for `mole`, require
    the mole conda env) and is much faster.
  - NaN/inf are median-imputed (not row-dropped) so all base learners share one
    aligned row index for the OOF matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, ClassVar

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_tools.filters import _bulk_tanimoto
from data_tools.inputs import featurize
from data_tools.load import butina_kfold_indices
from models.base import MetaModel, PXRModel
from models.delta_model import DeltaModel
from models.knn import KNN
from models.tabicl_model import TabICL
from models.xgboost_model import XGBoost

# Each base learner: (label, factory, representation list). The factory returns a
# fresh PXRModel so every fold/refit starts unfitted. Curated via score-table on
# the unblinded split (strong + mutually decorrelated families).
BaseSpec = tuple[str, Callable[[], PXRModel], list[str]]
DEFAULT_ROSTER: list[BaseSpec] = [
    ("chemprop", XGBoost, ["chemprop_finetuned"]),  # 2D-GNN pretrained (best single)
    ("unimol", XGBoost, ["unimol_finetuned"]),  # 3D shape
    ("mole", XGBoost, ["mole"]),  # pretrained tabular (diff family)
    ("tabular", XGBoost, ["rdkit", "jazzy", "xtb", "crystal_similarity"]),  # descriptors + QM + binder-resemblance
    ("tabicl", TabICL, ["rdkit"]),  # tabular foundation model
    ("knn", KNN, ["morgan"]),  # local Tanimoto
    ("delta", DeltaModel, ["rdkit"]),  # relational; also supplies a coverage meta-feature
]

DEFAULT_TARGET = "pEC50"
DEFAULT_N_SPLITS = 5


def _impute(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace non-finite entries (NaN/inf) with per-column medians."""
    out = X.astype(np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.take(medians, np.where(bad)[1])
    return out


class StackedEnsemble(MetaModel):
    """OOF-stacked ensemble with an ElasticNetCV meta-learner."""

    name: ClassVar[str] = "stacked_ensemble"

    def __init__(
        self,
        roster: list[BaseSpec] | None = None,
        n_splits: int = DEFAULT_N_SPLITS,
        target: str = DEFAULT_TARGET,
    ) -> None:
        self._roster = roster if roster is not None else DEFAULT_ROSTER
        self._n_splits = n_splits
        self._target = target
        self._fitted: list[PXRModel] = []  # base learners refit on full train
        self._medians: list[np.ndarray] = []  # per-learner imputation medians
        self._meta: Pipeline | None = None
        self._train_morgan: np.ndarray | None = None  # for NN-distance meta-feature
        self._delta_idx = next((i for i, (lab, _, _) in enumerate(self._roster) if lab == "delta"), None)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _featurize_all(self, df: pl.DataFrame, cache_dir: Path) -> list[np.ndarray]:
        """Featurize every base learner's representation block for df (raw, un-imputed)."""
        return [featurize(df, reps, cache_dir).astype(np.float64) for _, _, reps in self._roster]

    def _meta_features(self, base_preds: np.ndarray, nn_sim: np.ndarray) -> np.ndarray:
        """Assemble [base preds | NN similarity | disagreement | delta coverage flag]."""
        disagreement = base_preds.std(axis=1, keepdims=True)
        cols = [base_preds, nn_sim.reshape(-1, 1), disagreement]
        if self._delta_idx is not None:
            coverage = (base_preds[:, self._delta_idx] != 0.0).astype(np.float64).reshape(-1, 1)
            cols.append(coverage)
        return np.hstack(cols)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train_df: pl.DataFrame, cache_dir: Path) -> None:
        y = train_df[self._target].to_numpy().astype(np.float64)
        n = len(train_df)

        # Featurize once on full train; compute + store imputation medians.
        raw = self._featurize_all(train_df, cache_dir)
        self._medians = [np.nan_to_num(np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)) for X in raw]
        mats = [_impute(X, m) for X, m in zip(raw, self._medians)]

        self._train_morgan = featurize(train_df, "morgan", cache_dir).astype(np.float64)

        # OOF predictions via Butina grouped K-fold (row-sliced, no re-featurize).
        folds = butina_kfold_indices(train_df, n_splits=self._n_splits)
        oof = np.zeros((n, len(self._roster)), dtype=np.float64)
        nn_sim = np.zeros(n, dtype=np.float64)
        for f, val_idx in enumerate(folds):
            val = np.asarray(val_idx, dtype=int)
            if val.size == 0:
                continue
            tr = np.setdiff1d(np.arange(n), val, assume_unique=False)
            nn_sim[val] = _bulk_tanimoto(self._train_morgan[val], self._train_morgan[tr])
            for j, (label, factory, _) in enumerate(self._roster):
                model = factory()
                model.fit(mats[j][tr], y[tr])
                oof[val, j] = model.predict(mats[j][val])
            print(f"[stacked_ensemble] OOF fold {f + 1}/{self._n_splits} done ({val.size} val rows)")

        # Persist OOF predictions + meta-features for offline meta-learner tuning (e.g. Optuna).
        oof_path = Path("results/stacked_ensemble_oof.npz")
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            oof_path,
            oof=oof,
            y=y,
            nn_sim=nn_sim,
            labels=np.array([lab for lab, _, _ in self._roster]),
        )
        print(f"[stacked_ensemble] saved OOF predictions to {oof_path}")

        # Refit each base learner on the full training set so predict() has a model
        # to call on unseen rows (separate from the OOF predictions saved above).
        self._fitted = []
        for label, factory, _ in self._roster:
            model = factory()
            j = len(self._fitted)
            model.fit(mats[j], y)
            self._fitted.append(model)
            print(f"[stacked_ensemble] refit {label} on full train ({n} rows)")

        # Fit the ElasticNetCV meta-learner on OOF + meta-features only.
        Z = self._meta_features(oof, nn_sim)
        self._meta = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, max_iter=5000, random_state=42)),
            ]
        )
        self._meta.fit(Z, y)
        self._report_weights()

    def _report_weights(self) -> None:
        """Print meta-learner coefficients (base learners + meta-features)."""
        assert self._meta is not None
        coefs = self._meta.named_steps["enet"].coef_
        names = [lab for lab, _, _ in self._roster] + ["nn_sim", "disagreement"]
        if self._delta_idx is not None:
            names.append("delta_coverage")
        alpha = self._meta.named_steps["enet"].alpha_
        print(f"[stacked_ensemble] meta-learner alpha={alpha:.4g} (on standardized inputs):")
        for name, c in zip(names, coefs):
            print(f"    {name:16s} {c:+.4f}")

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, df: pl.DataFrame, cache_dir: Path) -> np.ndarray:
        assert self._meta is not None and self._train_morgan is not None, "Call fit() before predict()"
        raw = self._featurize_all(df, cache_dir)
        mats = [_impute(X, m) for X, m in zip(raw, self._medians)]
        base_preds = np.column_stack([m.predict(mats[j]) for j, m in enumerate(self._fitted)])

        df_morgan = featurize(df, "morgan", cache_dir).astype(np.float64)
        nn_sim = _bulk_tanimoto(df_morgan, self._train_morgan)

        Z = self._meta_features(base_preds, nn_sim)
        return self._meta.predict(Z)
