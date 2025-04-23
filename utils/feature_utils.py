# utils/feature_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_datetime_column(start_date, start_time, freq, periods):
    start_datetime = datetime.combine(start_date, start_time)
    return pd.date_range(start=start_datetime, periods=periods, freq=freq)

def create_ratio_column(col_a, col_b):
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(col_b != 0, col_a / col_b, np.nan)
    return pd.Series(ratio, index=col_a.index)

def create_manual_bins(series, n_bins):
    return pd.cut(series, bins=n_bins, labels=[f"Bin_{i+1}" for i in range(n_bins)], include_lowest=True)

def create_time_bins(datetime_series, freq):
    """
    Bin datetime values into the specified time frequency.
    Args:
        datetime_series (pd.Series): A datetime column.
        freq (str): A pandas frequency string. Examples:
                    'T' = minute, 'H' = hour, 'D' = day, 'W' = week, 'M' = month
    Returns:
        pd.Series: A series of datetime bins.
    """
    return pd.to_datetime(datetime_series).dt.to_period(freq).dt.to_timestamp()