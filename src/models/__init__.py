"""Model registry for the OpenADMET PXR challenge.

To add a new model: create a module in this directory, subclass PXRModel,
set a unique `name` class variable, and add it here.
"""

from models.base import PXRModel
from models.baseline import MeanBaseline, MedianBaseline
from models.decision_tree import DecisionTree
from models.knn import KNN
from models.linear_regression import LinearRegression
from models.mlp import MLP
from models.symbolic_regression import SymbolicRegression
from models.tabpfn_model import TabPFN
from models.xgboost_model import XGBoost

REGISTRY: dict[str, type[PXRModel]] = {
    MeanBaseline.name: MeanBaseline,
    MedianBaseline.name: MedianBaseline,
    KNN.name: KNN,
    DecisionTree.name: DecisionTree,
    LinearRegression.name: LinearRegression,
    MLP.name: MLP,
    SymbolicRegression.name: SymbolicRegression,
    TabPFN.name: TabPFN,
    XGBoost.name: XGBoost,
}

__all__ = ["PXRModel", "REGISTRY"]
