"""
visualization.py
================
Generates and saves all project visualizations including heatmaps,
bar charts, trend lines, correlation matrices, and more.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

PALETTE = "viridis"
FIG_DPI = 150
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#1a1d2e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#333",
    "figure.titlesize": 14,
})


def _save(fig: plt.Figure, name: str) -> str:
    """Save figure to charts directory and close it."""
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [CHART] Saved → {name}")
    return path


# ── 1. Distribution histogram ──────────────────────────────────────────────────
def plot_unemployment_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Unemployment_Rate"], bins=40, kde=True,
                 color="#7c6af7", ax=ax)
    ax.set_title("Distribution of Unemployment Rate", fontsize=14, color="white")
    ax.set_xlabel("Unemployment Rate (%)")
    ax.set_ylabel("Frequency")
    return _save(fig, "01_unemployment_distribution.png")


# ── 2. National trend line ─────────────────────────────────────────────────────
def plot_national_trend(df: pd.DataFrame) -> str:
    trend = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(trend["Date"], trend["Unemployment_Rate"],
            color="#00d4ff", linewidth=2, label="Monthly Avg")
    ax.fill_between(trend["Date"], trend["Unemployment_Rate"],
                    alpha=0.2, color="#00d4ff")
    ax.set_title("National Monthly Unemployment Trend (2019–2023)", fontsize=14, color="white")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    return _save(fig, "02_national_trend.png")


# ── 3. State-wise bar chart (avg) ──────────────────────────────────────────────
def plot_statewise_bar(df: pd.DataFrame) -> str:
    state_avg = (df.groupby("State")["Unemployment_Rate"]
                 .mean().sort_values(ascending=False).reset_index())
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.barh(state_avg["State"], state_avg["Unemployment_Rate"],
                   color=plt.cm.viridis(
                       np.linspace(0.2, 0.9, len(state_avg))))
    ax.set_title("Average Unemployment Rate by State", fontsize=14, color="white")
    ax.set_xlabel("Avg Unemployment Rate (%)")
    ax.invert_yaxis()
    return _save(fig, "03_statewise_bar.png")


# ── 4. Top 10 highest ─────────────────────────────────────────────────────────
def plot_top10_highest(df: pd.DataFrame) -> str:
    top10 = (df.groupby("State")["Unemployment_Rate"]
             .mean().nlargest(10).sort_values())
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    ax.barh(top10.index, top10.values, color=colors)
    ax.set_title("Top 10 States – Highest Unemployment", fontsize=14, color="white")
    ax.set_xlabel("Avg Unemployment Rate (%)")
    for i, v in enumerate(top10.values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", color="white", fontsize=9)
    return _save(fig, "04_top10_highest.png")


# ── 5. Top 10 lowest ──────────────────────────────────────────────────────────
def plot_top10_lowest(df: pd.DataFrame) -> str:
    bot10 = (df.groupby("State")["Unemployment_Rate"]
             .mean().nsmallest(10).sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
    ax.barh(bot10.index, bot10.values, color=colors)
    ax.set_title("Top 10 States – Lowest Unemployment", fontsize=14, color="white")
    ax.set_xlabel("Avg Unemployment Rate (%)")
    for i, v in enumerate(bot10.values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", color="white", fontsize=9)
    return _save(fig, "05_top10_lowest.png")


# ── 6. Heatmap – month × year ─────────────────────────────────────────────────
def plot_monthly_heatmap(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(
        index="Month", columns="Year",
        values="Unemployment_Rate", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_yticklabels(month_labels, rotation=0)
    ax.set_title("Unemployment Rate Heatmap (Month × Year)", fontsize=14, color="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    return _save(fig, "06_monthly_heatmap.png")


# ── 7. Correlation matrix ─────────────────────────────────────────────────────
def plot_correlation_matrix(df: pd.DataFrame) -> str:
    num_cols = ["Unemployment_Rate", "Labour_Participation_Rate",
                "Employment_Rate", "Rolling_3M_Avg", "Rolling_6M_Avg", "MoM_Change"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Matrix", fontsize=14, color="white")
    return _save(fig, "07_correlation_matrix.png")


# ── 8. Box plot – by year ─────────────────────────────────────────────────────
def plot_boxplot_by_year(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 6))
    df.boxplot(column="Unemployment_Rate", by="Year",
               ax=ax, patch_artist=True,
               boxprops=dict(facecolor="#7c6af7", color="white"),
               medianprops=dict(color="#00d4ff", linewidth=2),
               whiskerprops=dict(color="white"),
               capprops=dict(color="white"),
               flierprops=dict(markerfacecolor="#ff6b6b", markersize=4))
    ax.set_title("Unemployment Rate Distribution by Year", color="white")
    plt.suptitle("")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unemployment Rate (%)")
    return _save(fig, "08_boxplot_year.png")


# ── 9. Seasonal pattern line ──────────────────────────────────────────────────
def plot_seasonal_pattern(df: pd.DataFrame) -> str:
    seasonal = df.groupby("Month")["Unemployment_Rate"].mean()
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, 13), seasonal.values, marker="o",
            color="#f7c948", linewidth=2)
    ax.fill_between(range(1, 13), seasonal.values, alpha=0.2, color="#f7c948")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_title("Seasonal Unemployment Pattern (Monthly Average)", color="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Unemployment Rate (%)")
    return _save(fig, "09_seasonal_pattern.png")


# ── 10. Year-wise bar chart ───────────────────────────────────────────────────
def plot_yearwise_bar(df: pd.DataFrame) -> str:
    year_avg = df.groupby("Year")["Unemployment_Rate"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if v == year_avg.max() else "#3498db" for v in year_avg.values]
    bars = ax.bar(year_avg.index.astype(str), year_avg.values, color=colors)
    ax.set_title("Year-wise Average Unemployment Rate", color="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Avg Unemployment Rate (%)")
    for bar, val in zip(bars, year_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1, f"{val:.1f}%",
                ha="center", va="bottom", color="white", fontsize=9)
    return _save(fig, "10_yearwise_bar.png")


# ── 11. Region-wise pie chart ─────────────────────────────────────────────────
def plot_region_pie(df: pd.DataFrame) -> str:
    region_avg = df.groupby("Region")["Unemployment_Rate"].mean()
    fig, ax = plt.subplots(figsize=(8, 8))
    wedge_props = {"width": 0.6, "edgecolor": "white", "linewidth": 2}
    ax.pie(region_avg.values, labels=region_avg.index,
           autopct="%1.1f%%", startangle=140,
           colors=plt.cm.Set3(np.linspace(0, 1, len(region_avg))),
           wedgeprops=wedge_props)
    ax.set_title("Avg Unemployment by Region", color="white", fontsize=14)
    return _save(fig, "11_region_pie.png")


# ── 12. Scatter – employment vs unemployment ──────────────────────────────────
def plot_scatter_employment(df: pd.DataFrame) -> str:
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        sample["Employment_Rate"], sample["Unemployment_Rate"],
        c=sample["Year"], cmap="plasma", alpha=0.5, s=15
    )
    plt.colorbar(scatter, ax=ax, label="Year")
    ax.set_title("Employment Rate vs Unemployment Rate", color="white")
    ax.set_xlabel("Employment Rate (%)")
    ax.set_ylabel("Unemployment Rate (%)")
    return _save(fig, "12_scatter_employment.png")


# ── 13. Multi-state trend comparison ─────────────────────────────────────────
def plot_multistate_trend(df: pd.DataFrame, top_n: int = 6) -> str:
    top_states = (df.groupby("State")["Unemployment_Rate"]
                  .mean().nlargest(top_n).index.tolist())
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.tab10
    for i, state in enumerate(top_states):
        sdf = df[df["State"] == state].groupby("Date")["Unemployment_Rate"].mean()
        ax.plot(sdf.index, sdf.values, label=state,
                color=cmap(i / top_n), linewidth=1.5)
    ax.set_title(f"Unemployment Trend – Top {top_n} Most Affected States", color="white")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, "13_multistate_trend.png")


# ── 14. Severity distribution ─────────────────────────────────────────────────
def plot_severity_bar(df: pd.DataFrame) -> str:
    if "Severity" not in df.columns:
        return ""
    sev_counts = df["Severity"].value_counts()
    colors = {"Low": "#2ecc71", "Moderate": "#f39c12",
               "High": "#e74c3c", "Critical": "#8e44ad"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sev_counts.index, sev_counts.values,
                  color=[colors.get(s, "gray") for s in sev_counts.index])
    ax.set_title("Unemployment Severity Distribution", color="white")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, sev_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 50, str(val),
                ha="center", color="white", fontsize=9)
    return _save(fig, "14_severity_distribution.png")


def generate_all_charts(df: pd.DataFrame) -> list:
    """
    Run all visualization functions and return list of saved file paths.

    Parameters
    ----------
    df : pd.DataFrame
        Full processed DataFrame.

    Returns
    -------
    list of str
        Paths to all generated chart images.
    """
    print("[VIZ] Generating all charts...")
    paths = []
    funcs = [
        plot_unemployment_distribution,
        plot_national_trend,
        plot_statewise_bar,
        plot_top10_highest,
        plot_top10_lowest,
        plot_monthly_heatmap,
        plot_correlation_matrix,
        plot_boxplot_by_year,
        plot_seasonal_pattern,
        plot_yearwise_bar,
        plot_region_pie,
        plot_scatter_employment,
        plot_multistate_trend,
        plot_severity_bar,
    ]
    for fn in funcs:
        try:
            p = fn(df)
            if p:
                paths.append(p)
        except Exception as e:
            print(f"  [WARN] {fn.__name__} failed: {e}")
    print(f"[VIZ] {len(paths)} charts generated.")
    return paths
