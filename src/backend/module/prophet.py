import pandas as pd
from typing import Optional
from prophet import Prophet


def convert_df_for_prophet(
    data: pd.DataFrame,
    target_column_name: str,
    date_column_name: str,
    *event_column_name: Optional[str]
) -> pd.DataFrame:
    """prophetの時系列分析ができるようデータ整形を行う
    args:
      data: データフレーム
      target_column_name: 目的変数名
      date_column_name: 日付カラム名
      event_column_name: イベントカラム名（任意指定）
    return:
      prophet_df: prophetに適したデータ（カラム名変換、eventカラムをダミーデータ変換）
    """
    event_values = None
    if event_column_name:
        event_values = pd.get_dummies(
            data[event_column_name], drop_first=True, prefix="events"
        )

    # Prophetに合わせて、売上カラムをyに、日付カラムをdsに変換する必要がある
    data = data.rename(columns={target_column_name: "y", date_column_name: "ds"})
    prophet_df = pd.concat([data, event_values], axis=1)

    return prophet_df


def create_prophet_model(holiday_df: pd.DataFrame) -> Prophet:
    """prophetモデルを作成
    args:
      holiday_df:
        祝日データ
    return:
      prophet_model:
        prophetモデル（詳細はパラメーターを参照）
        https://facebook.github.io/prophet/docs/quick_start.html
    """
    prophet_model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, holidays=holiday_df
    )

    prophet_model.add_regressor(name="weekend")

    return prophet_model


def fit_predict_prophet_model(
    holiday_df: pd.DataFrame, prophet_df: pd.DataFrame
) -> (pd.DataFrame, Prophet):
    """
    モデルfitとpredictを実行する
    prophet_model.add_regressorをした場合は、そのカラムをfitとpredictに追加する
    """
    prophet_model = create_prophet_model(holiday_df)
    prophet_model.fit(prophet_df[["ds", "y", "weekend"]])
    pred = prophet_model.predict(prophet_df[["ds", "y", "weekend"]])
    return pred, prophet_model


def extract_df_with_prophet_data(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    予測値・説明変数が入ったデータフレームに、prophetデータを加える
    args:
      pred:
        Prophetで予測したデータ
      df:
        予測値・説明変数が入ったデータフレーム
    """
    col_from_prophet = [
        col
        for col in pred.columns
        if col in ["trend", "holidays", "yearly", "weekly", "extra_regressors_additive"]
    ]

    df_with_prophet = df.copy()
    for col in col_from_prophet:
        df_with_prophet[col] = pred[col]
    return df_with_prophet.rename(columns={"extra_regressors_additive": "weekend"})
