"""Model registry for the OpenADMET PXR challenge.

To add a new model: create a module in this directory, subclass PXRModel,
set a unique `name` class variable, and add it here.
"""

from models.base import PXRModel
from models.baseline import MeanBaseline, MedianBaseline

REGISTRY: dict[str, type[PXRModel]] = {
    MeanBaseline.name: MeanBaseline,
    MedianBaseline.name: MedianBaseline,
}

__all__ = ["PXRModel", "REGISTRY"]
