import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from backend.domain.random_forest_regressor import AbstractRandomForestRegressor


def split_to_train_test(
    df: pd.DataFrame, split_model: TimeSeriesSplit, target_col: str, features: list[str]
) -> tuple:
    for train_index, test_index in split_model.split(df):
        x_train = df.iloc[train_index][features]
        y_train = df[target_col].values[train_index]

        x_test = df.iloc[test_index][features]
        y_test = df[target_col].values[test_index]
    return x_train, y_train, x_test, y_test


class Regressor(AbstractRandomForestRegressor):
    def __init__(self):
        pass

    def make_random_forest_model(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        ccp_alpha: float,
        bootstrap: bool = True,
        random_state: int = 123,
    ) -> RandomForestRegressor:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "random_state": random_state,
            "ccp_alpha": ccp_alpha,
        }
        rf = RandomForestRegressor(**params)
        return rf

    def fit_predict_rf_model(
        self,
        rf_model: RandomForestRegressor,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
    ) -> (RandomForestRegressor, np.array):
        rf_model.fit(x_train, y_train)
        pred = rf_model.predict(x_test)
        return rf_model, np.ceil(pred)

    def calc_r2_score(self, y_true: np.array, y_pred: np.array) -> float:
        return r2_score(y_true, y_pred)
