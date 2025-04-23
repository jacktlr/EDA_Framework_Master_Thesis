import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def impute_missing(series: pd.Series, method: str, constant_value=None, df_context=None, window_size=3, n_neighbors=5):
    imputed_series = series.copy()

    if method == "Mean":
        imputed_series.fillna(series.mean(), inplace=True)

    elif method == "Median":
        imputed_series.fillna(series.median(), inplace=True)

    elif method == "Mode":
        imputed_series.fillna(series.mode().iloc[0], inplace=True)

    elif method == "Constant":
        imputed_series.fillna(constant_value, inplace=True)

    elif method == "Unknown":
        imputed_series.fillna("Unknown", inplace=True)

    elif method == "KNN":

        if df_context is None:
            raise ValueError("KNN imputation requires a full DataFrame context.")

        numeric_df = df_context.select_dtypes(include=["number"])

        if series.name not in numeric_df.columns:
            raise ValueError(f"Column `{series.name}` must be numeric for KNN imputation.")

        imputer = KNNImputer(n_neighbors=n_neighbors)

        df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

        imputed_series = pd.Series(df_imputed[series.name], index=series.index)

    elif method == "Interpolation":
        imputed_series = series.interpolate(method='linear', limit_direction='both')

    elif method == "Rolling Mean":
        imputed_series = series.copy()
        rolling_mean = series.rolling(window=window_size, min_periods=1).mean()
        imputed_series.fillna(rolling_mean, inplace=True)

    else:
        raise ValueError(f"Unsupported imputation method: {method}")

    return imputed_series


def get_imputation_methods(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return ["Mean", "Median", "Mode", "Constant", "KNN", "Interpolation", "Rolling Mean"]
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return ["Mode", "Constant", "Unknown"]
    else:
        return []
