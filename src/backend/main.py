import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

from streamlit.runtime.uploaded_file_manager import UploadedFile

sys.path.append(str(Path().absolute()))
from src.backend.module.prophet import (
    convert_df_for_prophet,
    fit_predict_prophet_model,
    extract_df_with_prophet_data,
)
from src.backend.module.read_dataset import read_dataset, get_holidays_jp
from src.backend.random_forest import regressor


FEATURES = ["trend", "weekend", "holidays", "weekly", "yearly"]


def handle_dataset(data: UploadedFile, date_col: str) -> (pd.DataFrame, pd.DataFrame):
    df = read_dataset(data, date_col)
    # TODO:動的に引数を渡せるようにする
    holidays_df = get_holidays_jp(2018, 1, 8, 2018, 1, 25)
    return df, holidays_df


def prophet(
    df: pd.DataFrame, holidays_df: pd.DataFrame, target_col: str, date_col: str
) -> (Prophet, pd.DataFrame, pd.DataFrame):
    prophet_df = convert_df_for_prophet(df, target_col, date_col)
    pred, prophet_model = fit_predict_prophet_model(holidays_df, prophet_df)
    df_with_prophet = extract_df_with_prophet_data(pred, df)
    return prophet_model, df_with_prophet, pred


def pred_random_forest(
    df_with_prophet: pd.DataFrame, target_col: str
) -> (RandomForestRegressor, pd.DataFrame, float):
    split_model = TimeSeriesSplit(n_splits=3, test_size=5)
    x_train, y_train, x_test, y_test = regressor.split_to_train_test(
        df_with_prophet, split_model, target_col, FEATURES
    )
    rf = regressor.Regressor()
    model = rf.make_random_forest_model(20, 9, 4, 8, 0.16)
    rf_model, pred = rf.fit_predict_rf_model(model, x_train, y_train, x_test)
    r2_score = rf.calc_r2_score(y_test, pred)
    return rf_model, x_train, r2_score
