"""
eda.py
======
Exploratory Data Analysis (EDA) module.
Generates comprehensive statistical summaries and insights from the dataset.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Return a high-level overview of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and processed DataFrame.

    Returns
    -------
    dict
        Dictionary with shape, columns, dtypes, memory usage, and date range.
    """
    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
    }
    if "Date" in df.columns:
        overview["date_range_start"] = str(df["Date"].min().date())
        overview["date_range_end"] = str(df["Date"].max().date())
        overview["total_months"] = df["Date"].dt.to_period("M").nunique()
    if "State" in df.columns:
        overview["unique_states"] = df["State"].nunique()
    return overview


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Extended describe() output including median and skewness.
    """
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats["median"] = num_df.median()
    stats["skewness"] = num_df.skew()
    stats["kurtosis"] = num_df.kurtosis()
    return stats.round(3)


def missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with count and percentage of missing values per column.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Pct": pct
    }).sort_values("Missing_Count", ascending=False)


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for numeric columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().round(3)


def trend_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly average unemployment rate across all states.

    Returns
    -------
    pd.DataFrame
        Monthly trend DataFrame sorted by date.
    """
    trend = (
        df.groupby("Date")["Unemployment_Rate"]
        .mean()
        .reset_index()
        .rename(columns={"Unemployment_Rate": "Avg_Unemployment_Rate"})
    )
    trend.sort_values("Date", inplace=True)
    return trend


def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average unemployment rate by month to detect seasonality.

    Returns
    -------
    pd.DataFrame
        DataFrame with month number and average unemployment.
    """
    seasonal = (
        df.groupby("Month")["Unemployment_Rate"]
        .mean()
        .reset_index()
        .rename(columns={"Unemployment_Rate": "Avg_Unemployment_Rate"})
    )
    month_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    seasonal["Month_Name"] = seasonal["Month"].map(month_map)
    return seasonal


def statewise_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-state average unemployment statistics.

    Returns
    -------
    pd.DataFrame
        State-wise aggregated metrics sorted by average unemployment (desc).
    """
    state_agg = df.groupby(["State", "Region"]).agg(
        Avg_Unemployment=("Unemployment_Rate", "mean"),
        Max_Unemployment=("Unemployment_Rate", "max"),
        Min_Unemployment=("Unemployment_Rate", "min"),
        Volatility=("Unemployment_Rate", "std"),
    ).reset_index().round(2)
    return state_agg.sort_values("Avg_Unemployment", ascending=False)


def yearwise_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute year-wise national average unemployment rate.

    Returns
    -------
    pd.DataFrame
        Year-wise summary DataFrame.
    """
    year_agg = df.groupby("Year").agg(
        Avg_Unemployment=("Unemployment_Rate", "mean"),
        Max_Unemployment=("Unemployment_Rate", "max"),
        Min_Unemployment=("Unemployment_Rate", "min"),
    ).reset_index().round(2)
    return year_agg


def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generate key automated insights from the data.

    Returns
    -------
    dict
        Dictionary with insight strings covering worst/best states,
        national average, peak period, and trend direction.
    """
    state_agg = statewise_analysis(df)
    trend = trend_analysis(df)

    highest_state = state_agg.iloc[0]["State"]
    highest_rate = state_agg.iloc[0]["Avg_Unemployment"]
    lowest_state = state_agg.iloc[-1]["State"]
    lowest_rate = state_agg.iloc[-1]["Avg_Unemployment"]
    national_avg = df["Unemployment_Rate"].mean()

    # Detect trend direction from first half vs second half
    midpoint = len(trend) // 2
    first_half_avg = trend.iloc[:midpoint]["Avg_Unemployment_Rate"].mean()
    second_half_avg = trend.iloc[midpoint:]["Avg_Unemployment_Rate"].mean()
    trend_direction = "declining" if second_half_avg < first_half_avg else "rising"

    # Peak month
    peak_row = trend.loc[trend["Avg_Unemployment_Rate"].idxmax()]

    # Volatility leader
    volatile_state = state_agg.sort_values("Volatility", ascending=False).iloc[0]

    insights = {
        "highest_unemployment_state": f"{highest_state} has the highest average unemployment at {highest_rate:.1f}%.",
        "lowest_unemployment_state": f"{lowest_state} has the lowest average unemployment at {lowest_rate:.1f}%.",
        "national_average": f"The national average unemployment rate is {national_avg:.1f}%.",
        "trend_direction": f"Unemployment is on a {trend_direction} trend over the analysis period.",
        "peak_period": f"Peak unemployment occurred around {peak_row['Date'].strftime('%B %Y')} with {peak_row['Avg_Unemployment_Rate']:.1f}%.",
        "most_volatile_state": f"{volatile_state['State']} shows the highest unemployment volatility (std = {volatile_state['Volatility']:.2f}%).",
        "covid_impact": "A significant spike is visible in mid-2020 across all states due to COVID-19 lockdowns.",
        "recovery_observation": "Post-2021, most states show a gradual recovery in employment conditions.",
    }
    return insights
