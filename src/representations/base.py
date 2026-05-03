"""Abstract base class for molecular representations."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import polars as pl


class Representation(ABC):
    """Transforms a Series of SMILES strings into a feature matrix."""

    name: ClassVar[str] = "base"

    @abstractmethod
    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_features) float64 array."""
