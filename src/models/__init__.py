"""Model registry for the OpenADMET PXR challenge.

To add a new model: create a module in this directory, subclass PXRModel,
set a unique `name` class variable, and add it here.
"""

from models.base import PXRModel, PXRPreConfigModel
from models.baseline import MeanBaseline, MedianBaseline
from models.decision_tree import DecisionTree
from models.knn import KNN
from models.linear_regression import LinearRegression
from models.mlp import MLP
from models.symbolic_regression import SymbolicRegression
from models.tabicl_model import TabICL
from models.tabpfn_model import TabPFN
from models.kan_model import KAN
from models.xgboost_model import XGBoost
from models.one_each_dim import OneofEachDim

REGISTRY: dict[str, type[PXRModel]] = {
    MeanBaseline.name: MeanBaseline,
    MedianBaseline.name: MedianBaseline,
    KNN.name: KNN,
    DecisionTree.name: DecisionTree,
    LinearRegression.name: LinearRegression,
    MLP.name: MLP,
    SymbolicRegression.name: SymbolicRegression,
    TabICL.name: TabICL,
    TabPFN.name: TabPFN,
    # KAN.name: KAN,
    XGBoost.name: XGBoost,
}

PRECONFIG_REGISTRY: dict[str, type[PXRPreConfigModel]] = {
    OneofEachDim.name: OneofEachDim,
}

__all__ = ["PXRModel", "REGISTRY"]
