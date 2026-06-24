"""
main.py
=======
Master orchestration script for the AI-Powered Unemployment Analysis System.

Run this file to execute the complete pipeline:
  python main.py

Pipeline Steps:
  1. Load & validate data
  2. Clean & preprocess data
  3. Feature engineering
  4. Exploratory data analysis
  5. Visualization generation
  6. ML model training & evaluation
  7. Forecasting
  8. Report generation
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_data, validate_data
from src.preprocessing import (
    clean_data, engineer_features,
    aggregate_state_data, save_processed_data
)
from src.eda import (
    dataset_overview, summary_statistics,
    generate_insights, trend_analysis, yearwise_analysis
)
from src.visualization import generate_all_charts
from src.feature_engineering import build_feature_matrix, get_X_y, split_data
from src.model_training import (
    train_and_evaluate, save_best_model,
    save_comparison_report
)
from src.forecasting import (
    national_forecast, plot_forecast,
    get_forecast_summary, save_forecast
)
from src.prediction import save_predictions
from src.utils import (
    ensure_dirs, print_banner,
    generate_markdown_report, save_json
)


def main():
    start_time = time.time()

    # ─── Setup ─────────────────────────────────────────────────────────────────
    print_banner("AI-POWERED UNEMPLOYMENT ANALYSIS SYSTEM")
    ensure_dirs()

    # ─── Step 1: Load Data ─────────────────────────────────────────────────────
    print_banner("STEP 1: DATA LOADING")
    df_raw = load_data()
    validation_report = validate_data(df_raw)

    # ─── Step 2: Preprocessing ─────────────────────────────────────────────────
    print_banner("STEP 2: DATA PREPROCESSING")
    df_clean = clean_data(df_raw)
    df = engineer_features(df_clean)
    save_processed_data(df)
    state_summary = aggregate_state_data(df)

    # ─── Step 3: EDA ──────────────────────────────────────────────────────────
    print_banner("STEP 3: EXPLORATORY DATA ANALYSIS")
    overview = dataset_overview(df)
    stats = summary_statistics(df)
    insights = generate_insights(df)

    print("\n📊 Dataset Overview:")
    for k, v in overview.items():
        print(f"   {k}: {v}")

    print("\n💡 Key Insights:")
    for _, text in insights.items():
        print(f"   • {text}")

    # ─── Step 4: Visualizations ───────────────────────────────────────────────
    print_banner("STEP 4: GENERATING VISUALIZATIONS")
    chart_paths = generate_all_charts(df)

    # ─── Step 5: ML Model Training ────────────────────────────────────────────
    print_banner("STEP 5: MACHINE LEARNING MODEL TRAINING")
    ml_df = build_feature_matrix(df)
    X, y = get_X_y(ml_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model_results, best_name, best_model, trained_models = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Save best model
    save_best_model(best_model, best_name, list(X.columns))
    save_comparison_report(model_results)

    # Save predictions
    y_pred = best_model.predict(X_test)
    save_predictions(y_test, y_pred)

    # ─── Step 6: Forecasting ──────────────────────────────────────────────────
    print_banner("STEP 6: UNEMPLOYMENT FORECASTING")
    forecast_df = national_forecast(df, steps=12)
    plot_forecast(df, forecast_df)
    forecast_summary = get_forecast_summary(forecast_df)
    save_forecast(forecast_df)

    print("\n📈 Forecast Summary:")
    print(f"   Next Month  ({forecast_summary['next_month']['date']}): "
          f"{forecast_summary['next_month']['rate']:.2f}%")
    print(f"   Next Quarter({forecast_summary['next_quarter']['date']}): "
          f"{forecast_summary['next_quarter']['rate']:.2f}%")
    print(f"   Next Year   ({forecast_summary['next_year']['date']}): "
          f"{forecast_summary['next_year']['rate']:.2f}%")

    # ─── Step 7: Reports ──────────────────────────────────────────────────────
    print_banner("STEP 7: GENERATING REPORTS")
    generate_markdown_report(overview, stats, insights, model_results, forecast_summary)
    save_json(insights, "insights.json")
    save_json(forecast_summary, "forecast_summary.json")
    save_json(model_results, "model_results.json")

    # ─── Done ─────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)
    print_banner(f"PIPELINE COMPLETE in {elapsed}s")
    print("\n✅ Outputs saved in: outputs/")
    print("   charts/       → 15 PNG visualizations")
    print("   models/       → Best trained model (.pkl)")
    print("   predictions/  → Forecast & test predictions (CSV)")
    print("   reports/      → project_report.md + JSON summaries")
    print("\n🚀 Launch dashboard: streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
