import pandas as pd
import numpy as np
from scipy.stats import zscore



def remove_outliers_zscore(series, threshold=3.0):
    clean_series = series.dropna()
    z_scores = zscore(clean_series)
    mask = np.abs(z_scores) <= threshold

    # Create a full mask aligned to the original index
    full_mask = pd.Series(False, index=series.index)
    full_mask[clean_series.index] = mask
    return series.where(full_mask)

def remove_outliers_iqr(series):
    clean_series = series.dropna()
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (clean_series >= lower) & (clean_series <= upper)

    full_mask = pd.Series(False, index=series.index)
    full_mask[clean_series.index] = mask
    return series.where(full_mask)

def quantile_clip(series, lower=0.01, upper=0.99):
    clean_series = series.dropna()
    q_low = clean_series.quantile(lower)
    q_high = clean_series.quantile(upper)
    mask = (clean_series >= q_low) & (clean_series <= q_high)

    full_mask = pd.Series(False, index=series.index)
    full_mask[clean_series.index] = mask
    return series.where(full_mask)

def mad_based_outlier_removal(series, threshold=3.5):
    clean_series = series.dropna()
    median = clean_series.median()
    mad = (np.abs(clean_series - median)).median()
    if mad == 0:
        return series
    modified_z = 0.6745 * (clean_series - median) / mad
    mask = np.abs(modified_z) < threshold

    full_mask = pd.Series(False, index=series.index)
    full_mask[clean_series.index] = mask
    return series.where(full_mask)

def manual_clip(series, min_val, max_val):
    # No dropna needed here — condition applies to full series
    return series.where((series >= min_val) & (series <= max_val))

def winsorize_series(series, lower_percentile=0.05, upper_percentile=0.95):
    # Safe without index reset — clip keeps full length
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower=lower, upper=upper)