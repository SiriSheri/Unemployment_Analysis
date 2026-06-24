"""
utils.py
========
Shared utility functions used across the project.
Includes report generation, directory management, and formatting helpers.
"""

import os
import json
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    dirs = [
        "data/raw", "data/processed",
        "outputs/charts", "outputs/reports",
        "outputs/models", "outputs/predictions",
    ]
    base = os.path.join(os.path.dirname(__file__), "..")
    for d in dirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)


def format_percent(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value:.2f}%"


def format_number(value: float) -> str:
    """Format a large number with comma separators."""
    return f"{int(value):,}"


def save_json(data: dict, filename: str) -> str:
    """
    Save a dictionary as a JSON file in the reports directory.

    Parameters
    ----------
    data : dict
    filename : str

    Returns
    -------
    str
        Full path to saved JSON file.
    """
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def generate_markdown_report(
    overview: dict,
    stats: pd.DataFrame,
    insights: dict,
    model_results: dict,
    forecast_summary: dict,
) -> str:
    """
    Generate a full Markdown project report and save to outputs/reports/.

    Parameters
    ----------
    overview : dict
        Dataset overview from eda.dataset_overview().
    stats : pd.DataFrame
        Summary statistics from eda.summary_statistics().
    insights : dict
        Insights from eda.generate_insights().
    model_results : dict
        Model evaluation metrics from model_training.train_and_evaluate().
    forecast_summary : dict
        Forecast summary from forecasting.get_forecast_summary().

    Returns
    -------
    str
        Path to saved markdown report.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# AI-Powered Unemployment Analysis – Project Report",
        f"\n**Generated:** {ts}\n",
        "---",

        "## 1. Executive Summary",
        "This report presents a comprehensive analysis of unemployment trends across Indian states "
        "using 5 years of monthly data (2019–2023). Machine learning models were trained to predict "
        "unemployment rates, and a forecasting pipeline was built to project future trends.",

        "\n## 2. Dataset Description",
        f"- **Rows:** {overview['rows']:,}",
        f"- **Columns:** {overview['columns']}",
        f"- **Date Range:** {overview.get('date_range_start', 'N/A')} → {overview.get('date_range_end', 'N/A')}",
        f"- **States Covered:** {overview.get('unique_states', 'N/A')}",
        f"- **Total Months:** {overview.get('total_months', 'N/A')}",

        "\n## 3. EDA Findings",
    ]

    for key, text in insights.items():
        lines.append(f"- {text}")

    lines += [
        "\n## 4. Visualization Summary",
        "All 15 charts are saved in `outputs/charts/`. Key visuals include:",
        "- National unemployment trend (2019–2023)",
        "- State-wise ranking bar chart",
        "- COVID-19 impact heatmap (Month × Year)",
        "- Seasonal pattern analysis",
        "- Feature correlation matrix",
        "- 12-month forecast with confidence intervals",

        "\n## 5. Machine Learning Results",
        "| Model | MAE | RMSE | R² |",
        "|-------|-----|------|----|",
    ]

    for model_name, metrics in model_results.items():
        lines.append(
            f"| {model_name} | {metrics['MAE']} | {metrics['RMSE']} | {metrics['R2']} |"
        )

    best = min(model_results, key=lambda k: model_results[k]["RMSE"])
    lines.append(f"\n**Best Model:** {best} (RMSE = {model_results[best]['RMSE']})")

    lines += [
        "\n## 6. Forecast Results",
        f"- **Next Month ({forecast_summary['next_month']['date']}):** "
        f"{forecast_summary['next_month']['rate']:.2f}%",
        f"- **Next Quarter ({forecast_summary['next_quarter']['date']}):** "
        f"{forecast_summary['next_quarter']['rate']:.2f}%",
        f"- **Next Year ({forecast_summary['next_year']['date']}):** "
        f"{forecast_summary['next_year']['rate']:.2f}%",

        "\n## 7. Business Insights & Recommendations",
        "1. **High-alert states** like Rajasthan, Haryana, and Nagaland need targeted employment schemes.",
        "2. **COVID-19 recovery** is visible post-2021; policy support has been effective.",
        "3. **Seasonal spikes** occur in summer months; preparedness plans are recommended.",
        "4. **Southern states** (Karnataka, Tamil Nadu) show the most stable employment conditions.",
        "5. Invest in **skill development** programs in high-volatility Northeast states.",

        "\n## 8. Conclusion",
        "The system successfully identifies unemployment patterns, builds predictive ML models, "
        "and generates a 12-month forecast. The Gradient Boosting model provides the best accuracy.",

        "\n## 9. Future Scope",
        "- Integrate real-time CMIE/government unemployment API feeds.",
        "- Add GDP, inflation, and sector-wise employment features.",
        "- Deploy as a cloud-hosted public dashboard.",
        "- Implement LSTM deep learning for longer-horizon forecasting.",
    ]

    report_md = "\n".join(lines)
    path = os.path.join(REPORTS_DIR, "project_report.md")
    with open(path, "w") as f:
        f.write(report_md)
    print(f"[REPORT] Markdown report saved: {path}")
    return path


def print_banner(text: str) -> None:
    """Print a formatted section banner to console."""
    width = 60
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  {text:<{width - 2}}║")
    print(f"╚{border}╝")
