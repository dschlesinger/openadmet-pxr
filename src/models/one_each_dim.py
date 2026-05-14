from models.base import PXRPreConfigModel
from models import XGBoost, TabICL, MLP
from representations import RDKitDescriptors, ChempropFingerprint, FinetunedChemeleonFingerprint, UniMolRepresentation
from models.utils import make_pca


from typing import ClassVar, List

import numpy as np
from sklearn.linear_model import LinearRegression

class OneofEachDim(PXRPreConfigModel):
    name: ClassVar[str] = 'OneofEachDim'
    required_repersentations: ClassVar[List[str]] = [RDKitDescriptors.name, ChempropFingerprint.name, FinetunedChemeleonFingerprint.name, UniMolRepresentation.name]

    xgboost = XGBoost()
    tabiclCP = TabICL()
    tabiclCh = TabICL()
    tabiclUM = TabICL()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.xgboost.fit(X, y)
        self.tabiclCP.fit(X, y)
        self.tabiclCh.fit(X, y)
        self.tabiclUM.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        a = self.xgboost.predict(X)
        b = self.tabiclCP.predict(X)
        c = self.tabiclCh.predict(X)
        d = self.tabiclUM.predict(X)
        return (a + b + c + d) / 4