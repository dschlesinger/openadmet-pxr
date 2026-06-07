"""Pairwise delta model with activity-cliff oversampling.

A Siamese network learns the *difference* in pEC50 between two molecules rather
than absolute potency. A shared MLP encoder ``f`` maps each molecule's feature
row to an embedding; a concat head ``g`` predicts ``delta = g(cat[f(x_i), f(x_j)])``.
The concat (non-separable) head can model pairwise interactions — activity cliffs,
where a small structural change causes a large potency change — so it contributes
signal orthogonal to ordinary absolute-pEC50 regressors.

At inference, each test molecule is anchored to its nearest training neighbors:

    pred(t) = mean over neighbors n of (y_n + delta(t, n))

with antisymmetry averaging ``delta(t, n) = 0.5 * (forward - reverse)``. Activity
cliff pairs (cosine >= SIM_CUTOFF and |dy| >= DELTA_THRESHOLD) are oversampled
during training. If a test molecule has fewer than ``MIN_NEIGHBORS`` training
neighbors within ``SIM_CUTOFF`` the model abstains and outputs 0 — the downstream
ensemble is expected to absorb these sentinels.

Adapted from ldbc1999 (rank 65), ``src/models/delta_model.py`` /
``src/data/cliff_analysis.py``. Their D-MPNN-over-SMILES encoder is replaced here
with an MLP over the input representation so the model stays a drop-in PXRModel
that consumes any registered representation; Tanimoto is replaced with cosine
similarity so it works on dense as well as binary fingerprints.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import PXRModel

# Training / inference constants (defaults follow ldbc1999 where applicable).
LR = 1e-3
BATCH_SIZE = 64
CLIFF_OVERSAMPLE = 3
K_NEIGHBORS = 10
SIM_CUTOFF = 0.7
MIN_NEIGHBORS = 5
DELTA_THRESHOLD = 1.0
SNAPSHOT_EPOCHS = 5
SEED = 42
_SIM_BLOCK = 1024  # row block size for batched cosine to bound memory


class _SiameseDelta(nn.Module):
    """Shared MLP encoder + concat delta head. Internal use only."""

    def __init__(self, in_features: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        e_i = self.encoder(x_i)
        e_j = self.encoder(x_j)
        return self.head(torch.cat([e_i, e_j], dim=1))  # (B, 1)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity matrix (n_a, n_b), batched over rows of ``a``."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.float32)
    for start in range(0, a.shape[0], _SIM_BLOCK):
        end = min(start + _SIM_BLOCK, a.shape[0])
        out[start:end] = a_norm[start:end] @ b_norm.T
    return out


class DeltaModel(PXRModel):
    """Pairwise delta regressor with activity-cliff oversampling and kNN anchoring."""

    name: ClassVar[str] = "delta"

    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        epochs: int = 60,
        n_pairs_per_epoch: int = 20_000,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._dropout = dropout
        self._epochs = epochs
        self._n_pairs_per_epoch = n_pairs_per_epoch
        self._preprocess: Pipeline = Pipeline(
            [
                ("vt", VarianceThreshold(0.0)),
                ("scaler", StandardScaler()),
            ]
        )
        self._net: Optional[_SiameseDelta] = None
        self._X_train: Optional[np.ndarray] = None  # raw features, for similarity
        self._X_train_scaled: Optional[np.ndarray] = None  # encoder input
        self._y_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Cliff detection + pair sampling
    # ------------------------------------------------------------------

    def _find_cliff_pairs(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return (n_cliff, 2) upper-triangle indices of activity-cliff pairs."""
        n = len(y)
        rows: list[int] = []
        cols: list[int] = []
        x_norm = X.astype(np.float32)
        x_norm = x_norm / (np.linalg.norm(x_norm, axis=1, keepdims=True) + 1e-12)
        for start in range(0, n, _SIM_BLOCK):
            end = min(start + _SIM_BLOCK, n)
            sim_block = x_norm[start:end] @ x_norm.T  # (block, n)
            for local_i, i in enumerate(range(start, end)):
                sims = sim_block[local_i]
                cand = np.where((sims >= SIM_CUTOFF) & (np.abs(y - y[i]) >= DELTA_THRESHOLD))[0]
                cand = cand[cand > i]  # upper triangle only
                rows.extend([i] * len(cand))
                cols.extend(cand.tolist())
        if not rows:
            return np.empty((0, 2), dtype=np.int64)
        return np.column_stack([rows, cols]).astype(np.int64)

    def _sample_pairs(self, n: int, rng: np.random.Generator, cliff_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample ordered (i, j) pairs for one epoch, oversampling cliff pairs."""
        budget = self._n_pairs_per_epoch
        if len(cliff_pairs) > 0:
            # Both orderings, tiled CLIFF_OVERSAMPLE times, capped at a quarter of the budget.
            ci = np.tile(np.concatenate([cliff_pairs[:, 0], cliff_pairs[:, 1]]), CLIFF_OVERSAMPLE)
            cj = np.tile(np.concatenate([cliff_pairs[:, 1], cliff_pairs[:, 0]]), CLIFF_OVERSAMPLE)
            n_cliff = min(len(ci), budget // 4)
            sel = rng.choice(len(ci), size=n_cliff, replace=False)
            fixed_i, fixed_j = ci[sel], cj[sel]
            budget -= n_cliff
        else:
            fixed_i = fixed_j = np.empty(0, dtype=np.int64)

        rand_i = rng.integers(0, n, size=budget)
        rand_j = rng.integers(0, n, size=budget)
        same = rand_i == rand_j
        rand_j[same] = (rand_j[same] + 1) % n

        all_i = np.concatenate([fixed_i, rand_i]).astype(np.int64)
        all_j = np.concatenate([fixed_j, rand_j]).astype(np.int64)
        return all_i, all_j

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)

        self._X_train = X.astype(np.float32)
        self._y_train = y.astype(np.float32)
        X_scaled: np.ndarray = self._preprocess.fit_transform(X).astype(np.float32)
        self._X_train_scaled = X_scaled

        cliff_pairs = self._find_cliff_pairs(self._X_train, self._y_train)
        print(f"DeltaModel: {len(cliff_pairs)} cliff pairs identified for {len(y)} compounds.")

        n = len(X_scaled)
        net = _SiameseDelta(X_scaled.shape[1], self._embedding_dim, self._dropout)
        X_tensor = torch.from_numpy(X_scaled)
        y_tensor = torch.from_numpy(self._y_train)

        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs, eta_min=LR * 0.01)
        criterion = nn.L1Loss()
        snapshots: list[dict] = []

        net.train()
        for epoch in range(self._epochs):
            idx_i, idx_j = self._sample_pairs(n, rng, cliff_pairs)
            targets = y_tensor[idx_i] - y_tensor[idx_j]
            perm = rng.permutation(len(idx_i))
            idx_i, idx_j, targets = idx_i[perm], idx_j[perm], targets[perm]
            for start in range(0, len(idx_i), BATCH_SIZE):
                bi = idx_i[start : start + BATCH_SIZE]
                bj = idx_j[start : start + BATCH_SIZE]
                if len(bi) < 2:
                    continue  # BatchNorm needs >1 sample
                optimizer.zero_grad()
                pred = net(X_tensor[bi], X_tensor[bj]).squeeze(1)
                loss = criterion(pred, targets[start : start + BATCH_SIZE])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            if SNAPSHOT_EPOCHS > 0 and epoch >= self._epochs - SNAPSHOT_EPOCHS:
                snapshots.append({k: v.clone() for k, v in net.state_dict().items()})

        if snapshots:
            avg_state = {k: torch.stack([sd[k].float() for sd in snapshots]).mean(0) for k in snapshots[0]}
            net.load_state_dict(avg_state)
        net.eval()
        self._net = net

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def _predict_deltas(self, x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
        """Batched delta prediction for paired (scaled) feature rows."""
        assert self._net is not None
        ti = torch.from_numpy(x_i.astype(np.float32))
        tj = torch.from_numpy(x_j.astype(np.float32))
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ti), BATCH_SIZE):
                bi = ti[start : start + BATCH_SIZE]
                bj = tj[start : start + BATCH_SIZE]
                out.append(self._net(bi, bj).squeeze(1).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() before predict()"
        assert self._X_train is not None and self._X_train_scaled is not None and self._y_train is not None

        X_scaled: np.ndarray = self._preprocess.transform(X).astype(np.float32)
        sim = _cosine_sim(X, self._X_train)  # (n_test, n_train), raw features
        n_test = len(X)

        # Flatten covered (test, neighbor) pairs into a single batched forward pass.
        pair_test_scaled: list[np.ndarray] = []
        pair_nbr_scaled: list[np.ndarray] = []
        neighbor_idx: list[np.ndarray] = []  # per covered test row
        covered_rows: list[int] = []
        for i in range(n_test):
            nbrs = np.where(sim[i] >= SIM_CUTOFF)[0]
            if len(nbrs) < MIN_NEIGHBORS:
                continue
            top = nbrs[np.argsort(-sim[i][nbrs])[:K_NEIGHBORS]]
            covered_rows.append(i)
            neighbor_idx.append(top)
            for nb in top:
                pair_test_scaled.append(X_scaled[i])
                pair_nbr_scaled.append(self._X_train_scaled[nb])

        preds = np.zeros(n_test, dtype=np.float32)
        if covered_rows:
            ti = np.asarray(pair_test_scaled, dtype=np.float32)
            tn = np.asarray(pair_nbr_scaled, dtype=np.float32)
            deltas_fwd = self._predict_deltas(ti, tn)
            deltas_rev = self._predict_deltas(tn, ti)
            deltas = 0.5 * (deltas_fwd - deltas_rev)  # antisymmetry averaging
            pos = 0
            for row, top in zip(covered_rows, neighbor_idx):
                k = len(top)
                d = deltas[pos : pos + k]
                preds[row] = float(np.mean(self._y_train[top] + d))
                pos += k

        n_covered = len(covered_rows)
        print(
            f"DeltaModel: {n_covered}/{n_test} test molecules covered "
            f"(>={MIN_NEIGHBORS} neighbors), {n_test - n_covered} abstained (pred=0)."
        )
        return preds
