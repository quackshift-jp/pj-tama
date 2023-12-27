import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


class AbstractRandomForestRegressor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_random_forest_model(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        ccp_alpha: float,
        bootstrap: bool,
        random_state: int,
    ):
        raise NotImplementedError()

    @abstractmethod
    def fit_predict_rf_model(
        self,
        rf_model: RandomForestRegressor,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> RandomForestRegressor:
        raise NotImplementedError()

    @abstractmethod
    def calc_r2_score(self, y_true: np.array, y_pred: np.array) -> float:
        raise NotImplementedError()
