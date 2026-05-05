"""Small dense neural network for the PXR challenge."""

from typing import ClassVar, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import PXRModel

DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (256, 128, 64)
DEFAULT_DROPOUT = 0.3
DEFAULT_MAX_EPOCHS = 300
DEFAULT_PATIENCE = 25
_VAL_FRACTION = 0.15
_LR = 1e-3
_WEIGHT_DECAY = 1e-4
_BATCH_SIZE = 64


class MLP(PXRModel):
    """Feed-forward network with BatchNorm, Dropout, and early stopping.

    Overfitting is controlled by three mechanisms:
      - Dropout after every hidden layer
      - L2 weight decay in the Adam optimizer
      - Early stopping on a held-out validation split with best-weight restore
    """

    name: ClassVar[str] = "mlp"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        dropout: float = DEFAULT_DROPOUT,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
    ) -> None:
        self._hidden_dims = hidden_dims
        self._dropout_p = dropout
        self._max_epochs = max_epochs
        self._patience = patience
        self._preprocess: Pipeline = Pipeline([
            ("vt", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
        ])
        self._net: Optional[nn.Module] = None

    def _build_net(self, in_features: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_features
        for width in self._hidden_dims:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(self._dropout_p)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr.astype(np.float32))

    def _val_loss(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            loss = nn.functional.mse_loss(self._net(X_val).squeeze(), y_val).item()
        self._net.train()
        return float(loss)

    def _train_loop(
        self,
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> None:
        assert self._net is not None
        opt = torch.optim.Adam(self._net.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=self._patience // 2, factor=0.5)
        best_loss, best_state, wait = float("inf"), None, 0
        n = len(X_tr)
        for _ in range(self._max_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, _BATCH_SIZE):
                idx = perm[start : start + _BATCH_SIZE]
                pred = self._net(X_tr[idx]).squeeze()
                loss = nn.functional.mse_loss(pred, y_tr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_loss = self._val_loss(X_val, y_val)
            scheduler.step(val_loss)
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self._patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s: np.ndarray = self._preprocess.fit_transform(X)
        n_val = max(1, int(len(X_s) * _VAL_FRACTION))
        idx = np.random.default_rng(42).permutation(len(X_s))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        self._net = self._build_net(X_s.shape[1])
        self._train_loop(
            self._to_tensor(X_s[tr_idx]), self._to_tensor(y[tr_idx]),
            self._to_tensor(X_s[val_idx]), self._to_tensor(y[val_idx]),
        )
        self._net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "Call fit() before predict()"
        X_s: np.ndarray = self._preprocess.transform(X)
        with torch.no_grad():
            return self._net(self._to_tensor(X_s)).squeeze().numpy()
