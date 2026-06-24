"""
dashboard/app.py
================
Streamlit interactive dashboard for the Unemployment Analysis System.

Launch with:
    streamlit run dashboard/app.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent to path so src imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unemployment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #252842);
        border: 1px solid #7c6af7;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #7c6af7; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .section-title {
        color: #7c6af7;
        border-bottom: 2px solid #7c6af7;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e, #252842);
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_processed_data():
    base = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(base, "data", "processed", "processed_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["Date"])
        return df
    # If pipeline hasn't run yet, generate on-the-fly
    from src.data_loader import load_data
    from src.preprocessing import clean_data, engineer_features
    df = engineer_features(clean_data(load_data()))
    return df


@st.cache_data
def load_json_report(filename):
    base = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(base, "outputs", "reports", filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_forecast():
    base = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(base, "outputs", "predictions", "national_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return pd.DataFrame()


@st.cache_data
def load_model_results():
    return load_json_report("model_results.json")


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_processed_data()
insights = load_json_report("insights.json")
forecast_summary = load_json_report("forecast_summary.json")
model_results = load_model_results()
forecast_df = load_forecast()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/statistics.png",
    width=60
)
st.sidebar.title("📊 Unemployment Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📋 Dataset Overview", "🗺️ State-wise Analysis",
     "📈 Trend Analysis", "🔮 Forecasting",
     "🤖 Model Performance", "💡 Insights"]
)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filters")

all_states = sorted(df["State"].unique())
selected_states = st.sidebar.multiselect(
    "Select States", all_states, default=all_states[:5]
)

all_years = sorted(df["Year"].unique())
selected_years = st.sidebar.multiselect(
    "Select Years", all_years, default=all_years
)

# Apply filters
filtered_df = df[
    (df["State"].isin(selected_states if selected_states else all_states)) &
    (df["Year"].isin(selected_years if selected_years else all_years))
]

st.sidebar.markdown("---")
# Download report button
report_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "reports", "project_report.md")
if os.path.exists(report_path):
    with open(report_path) as f:
        report_content = f.read()
    st.sidebar.download_button(
        "📥 Download Report",
        data=report_content,
        file_name="unemployment_report.md",
        mime="text/markdown"
    )

# ── Page: Home ────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🇮🇳 AI-Powered Unemployment Analysis System")
    st.markdown(
        "**Comprehensive analysis of unemployment trends across Indian states | 2019–2023**"
    )
    st.markdown("---")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    nat_avg = df["Unemployment_Rate"].mean()
    peak = df["Unemployment_Rate"].max()
    best_state = df.groupby("State")["Unemployment_Rate"].mean().idxmin()
    worst_state = df.groupby("State")["Unemployment_Rate"].mean().idxmax()

    col1.metric("🌐 National Avg", f"{nat_avg:.1f}%", help="Average across all states & months")
    col2.metric("📈 Peak Rate", f"{peak:.1f}%", help="Single highest recorded rate")
    col3.metric("🏆 Best State", best_state)
    col4.metric("⚠️ Worst State", worst_state)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📈 National Trend")
        trend = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()
        fig = px.line(trend, x="Date", y="Unemployment_Rate",
                      title="Monthly National Average Unemployment",
                      color_discrete_sequence=["#00d4ff"],
                      template="plotly_dark")
        fig.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("🗺️ State-wise Average")
        state_avg = df.groupby("State")["Unemployment_Rate"].mean().nlargest(15)
        fig2 = px.bar(
            state_avg.reset_index(),
            x="Unemployment_Rate", y="State",
            orientation="h",
            color="Unemployment_Rate",
            color_continuous_scale="Reds",
            title="Top 15 States by Average Unemployment",
            template="plotly_dark"
        )
        fig2.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                           yaxis={"autorange": "reversed"})
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔥 Unemployment Heatmap (All States)")
    pivot = df.pivot_table(index="State", columns="Year",
                           values="Unemployment_Rate", aggfunc="mean").round(1)
    fig3 = px.imshow(pivot, text_auto=True, color_continuous_scale="YlOrRd",
                     title="State × Year Unemployment Heatmap",
                     template="plotly_dark", aspect="auto")
    fig3.update_layout(paper_bgcolor="#1a1d2e")
    st.plotly_chart(fig3, use_container_width=True)


# ── Page: Dataset Overview ────────────────────────────────────────────────────
elif page == "📋 Dataset Overview":
    st.title("📋 Dataset Overview")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("States", df["State"].nunique())
    col3.metric("Months of Data", df["Date"].dt.to_period("M").nunique())

    st.markdown("### 🔍 Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### 📊 Summary Statistics")
    num_cols = ["Unemployment_Rate", "Labour_Participation_Rate", "Employment_Rate"]
    st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

    st.markdown("### 🚨 Missing Value Analysis")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
    st.dataframe(missing, use_container_width=True)

    st.markdown("### 📥 Download Processed Data")
    csv_data = df.to_csv(index=False)
    st.download_button("Download CSV", csv_data, "processed_data.csv", "text/csv")


# ── Page: State-wise Analysis ─────────────────────────────────────────────────
elif page == "🗺️ State-wise Analysis":
    st.title("🗺️ State-wise Analysis")
    st.markdown("---")

    state_agg = df.groupby(["State", "Region"]).agg(
        Avg=("Unemployment_Rate", "mean"),
        Max=("Unemployment_Rate", "max"),
        Min=("Unemployment_Rate", "min"),
        Std=("Unemployment_Rate", "std"),
    ).reset_index().round(2)
    state_agg.sort_values("Avg", ascending=False, inplace=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔴 Top 10 Highest Unemployment")
        top10 = state_agg.head(10)
        fig = px.bar(top10, x="Avg", y="State", orientation="h",
                     color="Avg", color_continuous_scale="Reds",
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
                          yaxis={"autorange": "reversed"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🟢 Top 10 Lowest Unemployment")
        bot10 = state_agg.tail(10).sort_values("Avg")
        fig2 = px.bar(bot10, x="Avg", y="State", orientation="h",
                      color="Avg", color_continuous_scale="Greens",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 State Summary Table")
    st.dataframe(state_agg, use_container_width=True)

    st.subheader("🌍 Region-wise Comparison")
    region_avg = df.groupby("Region")["Unemployment_Rate"].mean().reset_index()
    fig3 = px.pie(region_avg, names="Region", values="Unemployment_Rate",
                  hole=0.4, template="plotly_dark",
                  color_discrete_sequence=px.colors.qualitative.Set3)
    fig3.update_layout(paper_bgcolor="#1a1d2e")
    st.plotly_chart(fig3, use_container_width=True)

    # State deep dive
    st.subheader("🔎 State Deep Dive")
    selected = st.selectbox("Choose a state", all_states)
    sdf = df[df["State"] == selected].sort_values("Date")
    fig4 = px.line(sdf, x="Date", y="Unemployment_Rate",
                   title=f"Unemployment Trend – {selected}",
                   template="plotly_dark",
                   color_discrete_sequence=["#f7c948"])
    fig4.add_hrect(y0=0, y1=5, fillcolor="green", opacity=0.05)
    fig4.add_hrect(y0=10, y1=50, fillcolor="red", opacity=0.05)
    fig4.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
    st.plotly_chart(fig4, use_container_width=True)


# ── Page: Trend Analysis ──────────────────────────────────────────────────────
elif page == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")
    st.markdown("---")

    st.subheader("📅 Monthly National Trend")
    trend = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()
    fig = px.area(trend, x="Date", y="Unemployment_Rate",
                  title="National Monthly Unemployment Rate",
                  template="plotly_dark",
                  color_discrete_sequence=["#7c6af7"])
    fig.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📆 Year-wise Comparison")
        year_avg = df.groupby("Year")["Unemployment_Rate"].mean().reset_index()
        fig2 = px.bar(year_avg, x="Year", y="Unemployment_Rate",
                      color="Unemployment_Rate",
                      color_continuous_scale="Plasma",
                      text_auto=".1f",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🌀 Seasonal Pattern")
        seasonal = df.groupby("Month")["Unemployment_Rate"].mean().reset_index()
        months = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        seasonal["Month_Name"] = seasonal["Month"].apply(lambda m: months[m-1])
        fig3 = px.line(seasonal, x="Month_Name", y="Unemployment_Rate",
                       markers=True, template="plotly_dark",
                       color_discrete_sequence=["#f7c948"],
                       title="Average by Month (Seasonality)")
        fig3.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📊 Box Plot by Year")
    fig4 = px.box(df, x="Year", y="Unemployment_Rate",
                  color="Year", template="plotly_dark",
                  title="Unemployment Rate Distribution by Year")
    fig4.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
    st.plotly_chart(fig4, use_container_width=True)


# ── Page: Forecasting ─────────────────────────────────────────────────────────
elif page == "🔮 Forecasting":
    st.title("🔮 Unemployment Forecasting")
    st.markdown("---")

    if forecast_summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Next Month", f"{forecast_summary['next_month']['rate']:.2f}%",
                  help=forecast_summary['next_month']['date'])
        c2.metric("Next Quarter", f"{forecast_summary['next_quarter']['rate']:.2f}%",
                  help=forecast_summary['next_quarter']['date'])
        c3.metric("Next Year", f"{forecast_summary['next_year']['rate']:.2f}%",
                  help=forecast_summary['next_year']['date'])
    else:
        st.warning("Run main.py to generate forecasts.")

    if not forecast_df.empty:
        historical = df.groupby("Date")["Unemployment_Rate"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical["Date"], y=historical["Unemployment_Rate"],
            mode="lines", name="Historical",
            line=dict(color="#00d4ff", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"], y=forecast_df["Forecasted_Unemployment_Rate"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#ff6b6b", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([forecast_df["Upper_CI"], forecast_df["Lower_CI"][::-1]]),
            fill="toself", fillcolor="rgba(255,107,107,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval"
        ))
        fig.update_layout(
            title="Unemployment Rate: Historical + 12-Month Forecast",
            xaxis_title="Date", yaxis_title="Unemployment Rate (%)",
            template="plotly_dark",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            legend=dict(bgcolor="#1a1d2e")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Forecast Data Table")
        st.dataframe(forecast_df, use_container_width=True)
        csv = forecast_df.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV", csv,
                           "forecast.csv", "text/csv")


# ── Page: Model Performance ───────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 ML Model Performance")
    st.markdown("---")

    if model_results:
        results_df = pd.DataFrame(model_results).T.reset_index()
        results_df.rename(columns={"index": "Model"}, inplace=True)
        results_df.sort_values("RMSE", inplace=True)

        # Best model highlight
        best = results_df.iloc[0]
        st.success(f"🏆 Best Model: **{best['Model']}** | RMSE: {best['RMSE']} | R²: {best['R2']}")

        st.subheader("📊 Model Comparison")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(results_df, x="Model", y="RMSE",
                         color="RMSE", color_continuous_scale="Reds",
                         title="RMSE Comparison (lower is better)",
                         template="plotly_dark", text_auto=".4f")
            fig.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(results_df, x="Model", y="R2",
                          color="R2", color_continuous_scale="Greens",
                          title="R² Score Comparison (higher is better)",
                          template="plotly_dark", text_auto=".4f")
            fig2.update_layout(paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Full Metrics Table")
        st.dataframe(results_df, use_container_width=True)

        # Radar chart for model comparison
        categories = ["MAE", "MSE", "RMSE", "R2"]
        fig3 = go.Figure()
        for _, row in results_df.iterrows():
            fig3.add_trace(go.Scatterpolar(
                r=[float(row["MAE"]), float(row["MSE"]),
                   float(row["RMSE"]), float(row["R2"])],
                theta=categories,
                fill="toself",
                name=row["Model"]
            ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Model Performance Radar Chart",
            template="plotly_dark",
            paper_bgcolor="#1a1d2e"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Run main.py first to train models and generate results.")


# ── Page: Insights ────────────────────────────────────────────────────────────
elif page == "💡 Insights":
    st.title("💡 Automated Insights")
    st.markdown("---")

    if insights:
        for key, text in insights.items():
            icon = {
                "highest_unemployment_state": "🔴",
                "lowest_unemployment_state": "🟢",
                "national_average": "🌐",
                "trend_direction": "📈",
                "peak_period": "⚠️",
                "most_volatile_state": "🌊",
                "covid_impact": "🦠",
                "recovery_observation": "✅",
            }.get(key, "•")
            st.info(f"{icon} {text}")
    else:
        st.warning("Run main.py to generate insights.")

    st.markdown("---")
    st.subheader("📋 Business Recommendations")
    recommendations = [
        ("🎯", "High-alert states like Rajasthan, Haryana, and Nagaland need targeted employment programs."),
        ("📚", "Invest in skill development and vocational training in high-unemployment regions."),
        ("🌾", "Seasonal spikes in summer suggest need for agricultural employment safety nets."),
        ("🏙️", "Promote industrial decentralization to balance urban-rural employment gaps."),
        ("💰", "State-specific wage subsidies can accelerate recovery in post-COVID lagging states."),
        ("📡", "Integrate real-time CMIE data feeds for continuous monitoring."),
    ]
    for icon, rec in recommendations:
        st.success(f"{icon} {rec}")

    st.subheader("🔭 Future Scope")
    future = [
        "Integrate live unemployment API (CMIE, MOSPI) for real-time updates.",
        "Add GDP, inflation, and sector data as co-variates.",
        "Implement LSTM / Transformer for deep learning forecasting.",
        "Build state-level individual dashboards.",
        "Deploy on AWS/GCP with CI/CD pipeline.",
    ]
    for item in future:
        st.markdown(f"- {item}")
