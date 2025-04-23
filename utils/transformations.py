from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from scipy.stats import boxcox
import numpy as np
import pandas as pd

def apply_transformation(series, method):
    cleaned = series.dropna()

    if method == "Z-score Standardization":
        scaler = StandardScaler()
        result = pd.Series(scaler.fit_transform(cleaned.values.reshape(-1, 1)).flatten(), index=cleaned.index)

    elif method == "Min-Max Normalization":
        scaler = MinMaxScaler()
        result = pd.Series(scaler.fit_transform(cleaned.values.reshape(-1, 1)).flatten(), index=cleaned.index)

    elif method == "Log Transform":
        if (cleaned <= 0).any():
            raise ValueError("Log transform requires all values > 0.")
        result = np.log(cleaned)

    elif method == "Square Root Transform":
        if (cleaned < 0).any():
            raise ValueError("Square root requires non-negative values.")
        result = np.sqrt(cleaned)

    elif method == "Box-Cox Transform":
        if (cleaned <= 0).any():
            raise ValueError("Box-Cox requires all values > 0.")
        transformed, _ = boxcox(cleaned)
        result = pd.Series(transformed, index=cleaned.index)

    elif method == "Quantile Binning":
        binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        result = pd.Series(binner.fit_transform(cleaned.values.reshape(-1, 1)).flatten(), index=cleaned.index)

    else:
        raise ValueError("Unknown transformation.")

    return result