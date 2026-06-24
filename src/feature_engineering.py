"""
feature_engineering.py
=======================
Prepares final feature matrix (X) and target vector (y) for ML models.
Includes lag features, encoding, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature matrix for ML model training.

    Features used:
    - Temporal: Year, Month, Quarter
    - Rolling averages: 3-month, 6-month
    - Lag features: 1-month, 3-month, 6-month lags
    - Encoded categoricals: State (label-encoded), Region (label-encoded)
    - Labour Participation Rate

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed and feature-engineered DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for model training.
    """
    ml_df = df.copy().sort_values(["State", "Date"])

    # Lag features per state
    for lag in [1, 3, 6]:
        ml_df[f"Lag_{lag}M"] = (
            ml_df.groupby("State")["Unemployment_Rate"]
            .shift(lag)
        )

    # Drop rows with NaN lags (initial rows per state won't have lag data)
    ml_df.dropna(subset=["Lag_1M", "Lag_3M", "Lag_6M"], inplace=True)

    # Encode categorical columns
    le_state = LabelEncoder()
    le_region = LabelEncoder()
    ml_df["State_Encoded"] = le_state.fit_transform(ml_df["State"])
    ml_df["Region_Encoded"] = le_region.fit_transform(ml_df["Region"])

    return ml_df


def get_X_y(ml_df: pd.DataFrame) -> tuple:
    """
    Extract feature matrix X and target vector y.

    Parameters
    ----------
    ml_df : pd.DataFrame
        Feature matrix from build_feature_matrix().

    Returns
    -------
    tuple
        (X DataFrame, y Series)
    """
    feature_cols = [
        "Year", "Month", "Quarter",
        "Rolling_3M_Avg", "Rolling_6M_Avg",
        "Lag_1M", "Lag_3M", "Lag_6M",
        "Labour_Participation_Rate",
        "State_Encoded", "Region_Encoded",
    ]
    feature_cols = [c for c in feature_cols if c in ml_df.columns]
    X = ml_df[feature_cols]
    y = ml_df["Unemployment_Rate"]
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
        Proportion of data to use for testing (default 0.2).
    random_state : int

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, shuffle=False)
