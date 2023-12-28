import sys
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(str(Path().absolute()))
from backend.module.prophet import (
    convert_df_for_prophet,
    fit_predict_prophet_model,
    extract_df_with_prophet_data,
)
from backend.module.read_dataset import read_dataset, get_holidays_jp
from backend.random_forest import regressor


FEATURES = ["trend", "weekend", "holidays", "weekly", "yearly"]

df = read_dataset("./sample_data/白洋舎サンプルデータ.csv", "day")
holidays_df = get_holidays_jp(2018, 1, 8, 2018, 1, 25)

prophet_df = convert_df_for_prophet(df, "Yシャツ", "day")

pred, prophet_model = fit_predict_prophet_model(holidays_df, prophet_df)
df_with_prophet = extract_df_with_prophet_data(pred, df)

split_model = TimeSeriesSplit(n_splits=3, test_size=5)

x_train, y_train, x_test, y_test = regressor.split_to_train_test(
    df_with_prophet, split_model, "Yシャツ", FEATURES
)


rf = regressor.Regressor()

model = rf.make_random_forest_model(20, 9, 4, 8, 0.16)
rf_model, pred = rf.fit_predict_rf_model(model, x_train, y_train, x_test)
r2_score = rf.calc_r2_score(y_test, pred)
