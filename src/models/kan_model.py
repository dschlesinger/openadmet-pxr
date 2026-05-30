"""Kolmogorov-Arnold Network (KAN) model for the PXR challenge."""

from typing import ClassVar, Optional

import numpy as np
import torch
import torch.nn.functional as F
from kan import KAN as _KAN
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import PXRModel
from models.utils import make_pca

DEFAULT_HIDDEN_DIMS: tuple[int, ...] = (64, 32)
DEFAULT_GRID = 5
DEFAULT_K = 3
DEFAULT_STEPS = 300
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
# KAN parameter count scales as n_in * n_out * grid per layer; cap input dim
DEFAULT_MAX_INPUT_DIM = 128
# Update spline grid from data distribution this many times during training
_GRID_UPDATE_STEPS = 5


class KAN(PXRModel):
    """Kolmogorov-Arnold Network with B-spline edge activations.

    PCA (cuML GPU-accelerated when available) is applied when input exceeds
    max_input_dim to keep the first KAN layer tractable.
    Training uses a standard PyTorch Adam loop with periodic spline grid updates.
    """

    name: ClassVar[str] = "kan"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        grid: int = DEFAULT_GRID,
        k: int = DEFAULT_K,
        steps: int = DEFAULT_STEPS,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_input_dim: int = DEFAULT_MAX_INPUT_DIM,
    ) -> None:
        self._hidden_dims = hidden_dims
        self._grid = grid
        self._k = k
        self._steps = steps
        self._lr = lr
        self._batch_size = batch_size
        self._max_input_dim = max_input_dim
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._preprocess: Optional[Pipeline] = None
        self._net: Optional[_KAN] = None

    def _build_pipeline(self, n_samples: int, n_features: int) -> Pipeline:
        steps: list = [
            ("vt", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
        ]
        if n_features > self._max_input_dim:
            n_components = min(self._max_input_dim, n_samples)
            steps.append(("pca", make_pca(n_components)))
        return Pipeline(steps)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr.astype(np.float32), device=self._device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._preprocess = self._build_pipeline(X.shape[0], X.shape[1])
        X_s: np.ndarray = self._preprocess.fit_transform(X)
        X_t = self._to_tensor(X_s)
        y_t = self._to_tensor(y)

        width = [X_s.shape[1], *self._hidden_dims, 1]
        self._net = _KAN(
            width=width,
            grid=self._grid,
            k=self._k,
            seed=42,
            auto_save=False,
            device=self._device,
        )

        # Init grid from data distribution before training
        self._net.update_grid_from_samples(X_t)

        opt = torch.optim.Adam(self._net.parameters(), lr=self._lr)
        grid_update_interval = max(1, self._steps // _GRID_UPDATE_STEPS)
        n = len(X_t)

        for step in range(self._steps):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, self._batch_size):
                idx = perm[start:start + self._batch_size]
                pred = self._net(X_t[idx]).squeeze()
                loss = F.mse_loss(pred, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Periodically refit spline grid to current activation distribution
            if (step + 1) % grid_update_interval == 0 and step < self._steps - 1:
                with torch.no_grad():
                    self._net.update_grid_from_samples(X_t)

        self._net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None and self._preprocess is not None, "Call fit() before predict()"
        X_s: np.ndarray = self._preprocess.transform(X)
        with torch.no_grad():
            out = self._net(self._to_tensor(X_s)).squeeze()
            return out.cpu().numpy()
