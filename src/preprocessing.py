"""
preprocessing.py
================
Full data cleaning and preprocessing pipeline.
Handles missing values, outliers, type conversion, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import warnings
warnings.filterwarnings("ignore")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full data cleaning on the raw unemployment DataFrame.

    Steps:
    - Remove duplicate rows
    - Convert Date column to datetime
    - Handle missing values via forward-fill then median imputation
    - Clip extreme outliers using IQR method
    - Ensure correct data types

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    print("[PREPROCESSING] Starting data cleaning pipeline...")
    df = df.copy()

    # Step 1: Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  [+] Removed {before - len(df)} duplicate rows.")

    # Step 2: Parse dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        invalid_dates = df["Date"].isnull().sum()
        if invalid_dates > 0:
            print(f"  [!] Dropping {invalid_dates} rows with unparseable dates.")
            df.dropna(subset=["Date"], inplace=True)

    # Step 3: Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [+] Imputed {missing} missing values in '{col}' with median ({median_val:.2f}).")

    # Step 4: Treat outliers using IQR clipping
    for col in ["Unemployment_Rate", "Labour_Participation_Rate", "Employment_Rate"]:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            clipped = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            if clipped > 0:
                print(f"  [+] Clipped {clipped} outliers in '{col}'.")

    # Step 5: Ensure correct types
    if "State" in df.columns:
        df["State"] = df["State"].astype(str).str.strip()
    if "Region" in df.columns:
        df["Region"] = df["Region"].astype(str).str.strip()

    print(f"[PREPROCESSING] Cleaning complete. Final shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing columns to enhance ML model accuracy.

    New features include:
    - Year, Month, Quarter extracted from Date
    - Rolling averages (3-month, 6-month) per state
    - Month-over-month change in unemployment
    - Unemployment severity category

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    print("[FEATURE ENG] Engineering new features...")
    df = df.copy()
    df.sort_values(["State", "Date"], inplace=True)

    # Time-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Month_Name"] = df["Date"].dt.strftime("%B")
    df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)

    # Rolling statistics per state
    df["Rolling_3M_Avg"] = (
        df.groupby("State")["Unemployment_Rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(2)
    )
    df["Rolling_6M_Avg"] = (
        df.groupby("State")["Unemployment_Rate"]
        .transform(lambda x: x.rolling(6, min_periods=1).mean())
        .round(2)
    )

    # Month-over-month change
    df["MoM_Change"] = (
        df.groupby("State")["Unemployment_Rate"]
        .diff()
        .round(2)
    )

    # Severity category
    df["Severity"] = pd.cut(
        df["Unemployment_Rate"],
        bins=[0, 5, 10, 20, 100],
        labels=["Low", "Moderate", "High", "Critical"],
        right=False
    )

    print(f"[FEATURE ENG] Features added. Shape now: {df.shape}")
    return df


def aggregate_state_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate unemployment data at the state level across all time periods.

    Returns mean, max, min, and std of unemployment rate per state.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame.

    Returns
    -------
    pd.DataFrame
        State-level aggregated summary DataFrame.
    """
    agg = df.groupby("State").agg(
        Avg_Unemployment=("Unemployment_Rate", "mean"),
        Max_Unemployment=("Unemployment_Rate", "max"),
        Min_Unemployment=("Unemployment_Rate", "min"),
        Std_Unemployment=("Unemployment_Rate", "std"),
        Avg_Labour_Participation=("Labour_Participation_Rate", "mean"),
        Region=("Region", "first"),
    ).reset_index().round(2)

    agg.sort_values("Avg_Unemployment", ascending=False, inplace=True)
    return agg


def normalize_data(df: pd.DataFrame, method: str = "minmax") -> tuple:
    """
    Normalize/scale numerical columns in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    method : str
        'minmax' for MinMaxScaler or 'standard' for StandardScaler.

    Returns
    -------
    tuple
        (scaled_df, scaler) where scaled_df has normalized numerical columns.
    """
    numeric_cols = [
        "Unemployment_Rate", "Labour_Participation_Rate",
        "Employment_Rate", "Rolling_3M_Avg", "Rolling_6M_Avg"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df_scaled, scaler


def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.csv") -> str:
    """
    Save the processed DataFrame to the data/processed directory.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame.
    filename : str
        Output filename.

    Returns
    -------
    str
        Full path where the file was saved.
    """
    processed_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed"
    )
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, filename)
    df.to_csv(path, index=False)
    print(f"[SAVE] Processed data saved to: {path}")
    return path
