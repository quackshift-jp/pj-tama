import pandas as pd
import numpy as np

import shap
from sklearn.ensemble import RandomForestRegressor


def create_and_plot_shap_value(
    rf_model: RandomForestRegressor, x_train: pd.DataFrame
) -> np.ndarray:
    """
    モデルのSHAP値を算出する
    args:
      rf_model: 訓練済みのRandomForestモデル
      x_train: 訓練用の説明変数テーブル
    """
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X=x_train)
    shap.summary_plot(shap_values, x_train)
    return shap_values
