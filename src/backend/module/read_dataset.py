import pandas as pd
import datetime

import jpholiday
import holidays


def read_dataset(file, date_column_name: str) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=[date_column_name])
    df.columns = [
        col.lower() if col not in [date_column_name] else col for col in df.columns
    ]
    return df


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
