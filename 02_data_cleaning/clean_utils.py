import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


def analyze_missing(df):
    missing_percent = df.isnull().mean() * 100
    missing_df = pd.DataFrame(
        {"column": df.columns, "missing_percent": missing_percent}
    )
    missing_df = missing_df[missing_df["missing_percent"] > 0].sort_values(
        by="missing_percent", ascending=False
    )

    try:
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        if not cols_with_missing:
            mcar_result = "No missing values found, MCAR test not applicable."
        else:
            numeric_df_for_mcar = df[cols_with_missing].select_dtypes(include=np.number)
            if numeric_df_for_mcar.empty:
                mcar_result = "No numeric columns with missing values for MCAR test."
            else:
                mcar_result = "Visual analysis required (e.g., using missingno matrix) or use external MCAR test library."
    except Exception as e:
        mcar_result = f"MCAR test could not be performed: {e}"

    return missing_df, mcar_result


def detect_outliers(df, column, method="zscore", threshold=3, random_state=42):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    data = df[column].dropna()
    if data.empty:
        return pd.Series([False] * len(df), index=df.index)

    if method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        outlier_indices = data.index[z_scores > threshold]
    elif method == "isoforest":
        model = IsolationForest(contamination="auto", random_state=random_state)
        model.fit(data.values.reshape(-1, 1))
        outlier_preds = model.predict(data.values.reshape(-1, 1))
        outlier_indices = data.index[outlier_preds == -1]
    else:
        raise ValueError("Method must be 'zscore' or 'isoforest'")

    is_outlier = pd.Series(False, index=df.index)
    is_outlier.loc[outlier_indices] = True
    return is_outlier


def apply_imputer(df, method="median", k=5, random_state=42):
    df_imputed = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    if method == "median":
        imputer = SimpleImputer(strategy="median")
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=k)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=random_state)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    else:
        raise ValueError("Method must be 'median', 'knn', or 'mice'")

    return df_imputed


def apply_winsorizer(series, limits=(0.01, 0.99)):
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    return winsorize(series, limits=limits)


def apply_clipper(series, lower_quantile=0.01, upper_quantile=0.99):
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.clip(lower=lower_bound, upper=upper_bound)
