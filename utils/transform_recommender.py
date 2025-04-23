from scipy.stats import skew
import numpy as np
import pandas as pd


def recommend_transformation(series: pd.Series) -> str:
    """Return a recommendation based on the column's properties."""
    if series.isnull().all():
        return "Column is empty. No transformation recommended."

    cleaned = series.dropna()
    sk = skew(cleaned)
    std = np.std(cleaned)
    min_val = cleaned.min()
    max_val = cleaned.max()

    if std == 0:
        return "Drop or ignore — zero variance."
    elif sk > 1.5 and min_val > 0:
        return "Log or Box-Cox transformation (right skew)."
    elif sk < -1.5 and (cleaned + abs(min_val) + 1).min() > 0:
        return "Consider reflection + log (left skew)."
    elif min_val < 0:
        return "Standardization or robust scaling (contains negatives)."
    elif (max_val - min_val) > 10000:
        return "Min-Max normalization (large range)."
    else:
        return "No transformation strongly needed."