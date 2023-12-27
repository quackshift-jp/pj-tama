import sys
from pathlib import Path
import pandas as pd


sys.path.append(str(Path().absolute()))

from backend.module.prophet import (
    convert_df_for_prophet,
    fit_predict_prophet_model,
    extract_df_with_prophet_data,
)

from backend.module.read_dataset import read_dataset, get_holidays_jp

df = read_dataset("./sample_data/白洋舎サンプルデータ.csv", "day")
holidays_df = get_holidays_jp(2018, 1, 8, 2018, 1, 25)

prophet_df = convert_df_for_prophet(df, "Yシャツ", "day")

pred, prophet_model = fit_predict_prophet_model(holidays_df, prophet_df)
df_with_prophet = extract_df_with_prophet_data(pred, df)
