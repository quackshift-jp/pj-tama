import pandas as pd
import datetime

import jpholiday
import holidays


def attach_holiday_to_df(df: pd.DataFrame, date_column_name: str) -> pd.DataFrame:
    jp_holidays = holidays.Japan(years=[2018, 2019])
    df["weekend"] = df[date_column_name].apply(
        lambda x: 1 if x.weekday() >= 5 or x in jp_holidays else 0
    )
    return df


def calc_pivot(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    pivot_table = pd.pivot_table(
        df, index=index, columns=columns, aggfunc="count", fill_value=0
    ).reset_index()
    # TODO:カラム名を動的に決定するように修正する
    pivot_table.columns = ["day", "Yシャツ", "コート", "ズボン"]
    return pivot_table


def read_dataset(file, date_column_name: str) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=[date_column_name], encoding="cp932")
    df.columns = [
        col.lower() if col not in [date_column_name] else col for col in df.columns
    ]
    pivot_df = calc_pivot(df, index="day", columns="カテゴリ")
    return attach_holiday_to_df(pivot_df, "day")


def get_holidays_jp(
    from_year: int,
    from_month: int,
    from_day: int,
    to_year: int,
    to_month: int,
    to_day: int,
) -> pd.DataFrame:
    holiday = jpholiday.between(
        datetime.date(from_year, from_month, from_day),
        datetime.date(to_year, to_month, to_day),
    )
    ds = [date[0] for date in holiday]
    holiday_name = [date[1] for date in holiday]
    holiday_df = pd.DataFrame({"ds": ds, "holiday": holiday_name})
    holiday_df["ds"] = pd.to_datetime(holiday_df["ds"])
    return holiday_df
