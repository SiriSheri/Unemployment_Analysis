"""
data_loader.py
==============
Handles loading, downloading, and validating unemployment datasets.
Supports CSV files and generates synthetic data for demonstration purposes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


def generate_synthetic_unemployment_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic unemployment dataset for Indian states.

    This function creates multi-year monthly unemployment data across all
    major Indian states with realistic seasonal and trend patterns.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Date, State, Unemployment_Rate,
        Labour_Participation_Rate, Employment_Rate, Estimated_Employed,
        Estimated_Unemployed, Region.
    """
    np.random.seed(seed)

    states = {
        "Andhra Pradesh": ("South", 6.5),
        "Arunachal Pradesh": ("Northeast", 4.2),
        "Assam": ("Northeast", 8.1),
        "Bihar": ("East", 10.3),
        "Chhattisgarh": ("Central", 5.8),
        "Goa": ("West", 9.2),
        "Gujarat": ("West", 3.7),
        "Haryana": ("North", 18.5),
        "Himachal Pradesh": ("North", 7.4),
        "Jharkhand": ("East", 7.8),
        "Karnataka": ("South", 2.9),
        "Kerala": ("South", 11.4),
        "Madhya Pradesh": ("Central", 4.6),
        "Maharashtra": ("West", 6.2),
        "Manipur": ("Northeast", 12.6),
        "Meghalaya": ("Northeast", 5.4),
        "Mizoram": ("Northeast", 3.1),
        "Nagaland": ("Northeast", 20.1),
        "Odisha": ("East", 5.3),
        "Punjab": ("North", 7.9),
        "Rajasthan": ("North", 25.2),
        "Sikkim": ("Northeast", 2.3),
        "Tamil Nadu": ("South", 3.8),
        "Telangana": ("South", 5.7),
        "Tripura": ("Northeast", 8.9),
        "Uttar Pradesh": ("North", 4.4),
        "Uttarakhand": ("North", 4.1),
        "West Bengal": ("East", 5.6),
        "Delhi": ("North", 14.3),
        "Jammu & Kashmir": ("North", 12.8),
    }

    records = []
    # Generate 5 years of monthly data
    start_date = datetime(2019, 1, 1)

    for month_offset in range(60):  # 5 years = 60 months
        current_date = start_date + timedelta(days=month_offset * 30)

        # COVID shock: spike in mid-2020, recovery afterward
        covid_shock = 0.0
        months_since_start = month_offset
        if 12 <= months_since_start <= 18:
            covid_shock = np.random.uniform(5, 15)
        elif 18 < months_since_start <= 24:
            covid_shock = np.random.uniform(2, 8)

        # Seasonal factor: unemployment tends to rise in summer months
        seasonal_factor = 1.5 * np.sin(2 * np.pi * current_date.month / 12)

        # Gradual downward trend post-2021
        trend_factor = -0.02 * max(0, months_since_start - 24)

        for state, (region, base_rate) in states.items():
            # Compute final unemployment rate with noise
            rate = (
                base_rate
                + covid_shock
                + seasonal_factor
                + trend_factor
                + np.random.normal(0, 0.8)
            )
            rate = max(0.5, min(rate, 45.0))  # Clamp to realistic range

            # Derived metrics
            labour_participation = np.random.uniform(38, 55)
            employment_rate = labour_participation * (1 - rate / 100)
            pop_scale = np.random.uniform(500_000, 5_000_000)
            estimated_employed = int(employment_rate / 100 * pop_scale)
            estimated_unemployed = int(rate / 100 * labour_participation / 100 * pop_scale)

            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "State": state,
                "Unemployment_Rate": round(rate, 2),
                "Labour_Participation_Rate": round(labour_participation, 2),
                "Employment_Rate": round(employment_rate, 2),
                "Estimated_Employed": estimated_employed,
                "Estimated_Unemployed": estimated_unemployed,
                "Region": region,
            })

    return pd.DataFrame(records)


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load unemployment data from a CSV file or generate synthetic data.

    If a valid filepath is provided and the file exists, it loads from disk.
    Otherwise, synthetic data is generated and saved to the raw data folder.

    Parameters
    ----------
    filepath : str, optional
        Path to an existing CSV data file.

    Returns
    -------
    pd.DataFrame
        Loaded or generated unemployment DataFrame.
    """
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    default_path = os.path.join(raw_dir, "unemployment_data.csv")

    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
    elif os.path.exists(default_path):
        print(f"[INFO] Loading cached data from: {default_path}")
        df = pd.read_csv(default_path)
    else:
        print("[INFO] No dataset found. Generating synthetic unemployment data...")
        df = generate_synthetic_unemployment_data()
        df.to_csv(default_path, index=False)
        print(f"[INFO] Synthetic data saved to: {default_path}")

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate dataset quality and return a quality report.

    Checks for required columns, missing values, duplicate rows,
    and basic type correctness.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame to validate.

    Returns
    -------
    dict
        A dictionary with keys: shape, columns, missing_values,
        duplicates, dtypes, quality_score.
    """
    required_columns = [
        "Date", "State", "Unemployment_Rate",
        "Labour_Participation_Rate", "Employment_Rate"
    ]

    report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "required_columns_present": all(c in df.columns for c in required_columns),
    }

    # Compute a simple quality score (0-100)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100
    dup_penalty = min(report["duplicates"] / max(df.shape[0], 1) * 100, 20)
    report["quality_score"] = round(max(0, completeness - dup_penalty), 2)

    print(f"\n[VALIDATION] Dataset shape: {report['shape']}")
    print(f"[VALIDATION] Missing values: {sum(report['missing_values'].values())}")
    print(f"[VALIDATION] Duplicate rows: {report['duplicates']}")
    print(f"[VALIDATION] Quality score: {report['quality_score']}%")

    return report
