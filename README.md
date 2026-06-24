# 📊 AI-Powered Unemployment Analysis and Forecasting System

> A production-grade Python data analytics and machine learning project for analyzing,
> visualizing, and forecasting unemployment trends across Indian states.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📥 Data Generation | Realistic 5-year synthetic dataset (30 states × 60 months) |
| 🧹 Data Cleaning | Deduplication, outlier clipping, missing value imputation |
| 🔧 Feature Engineering | Lag features, rolling averages, severity labels |
| 📊 EDA | 10+ analytical summaries + automated insights |
| 📈 Visualizations | 15 publication-quality charts saved to `outputs/charts/` |
| 🤖 ML Models | Linear Regression, Decision Tree, Random Forest, Gradient Boosting |
| 🔮 Forecasting | 12-month national unemployment forecast with confidence intervals |
| 💻 Dashboard | 7-page interactive Streamlit dashboard |
| 📝 Reports | Auto-generated Markdown + JSON reports |

---

## 📁 Project Structure

```
Unemployment_Analysis/
│
├── data/
│   ├── raw/                    ← Raw / downloaded CSV data
│   └── processed/              ← Cleaned, feature-engineered data
│
├── src/
│   ├── data_loader.py          ← Load & validate data
│   ├── preprocessing.py        ← Cleaning, feature engineering
│   ├── eda.py                  ← Statistical analysis & insights
│   ├── visualization.py        ← 14 chart generation functions
│   ├── feature_engineering.py  ← ML feature matrix builder
│   ├── model_training.py       ← Train & evaluate 4 ML models
│   ├── forecasting.py          ← 12-month unemployment forecast
│   ├── prediction.py           ← Load model & make predictions
│   └── utils.py                ← Reports, JSON, directory helpers
│
├── dashboard/
│   └── app.py                  ← Streamlit 7-page dashboard
│
├── outputs/
│   ├── charts/                 ← 15 PNG charts
│   ├── reports/                ← project_report.md + JSON summaries
│   ├── models/                 ← Best model .pkl + metadata
│   └── predictions/            ← Forecast CSV + test predictions
│
├── main.py                     ← Master pipeline script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone or unzip the project
cd Unemployment_Analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1 – Run the full pipeline
```bash
python main.py
```

This will:
- Generate/load the dataset
- Clean and engineer features
- Train and evaluate 4 ML models
- Generate 15 visualizations
- Build a 12-month forecast
- Save all outputs and reports

### Step 2 – Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | KPI cards, national trend, heatmap |
| 📋 Dataset Overview | Raw data, statistics, missing values |
| 🗺️ State-wise Analysis | Top 10 highest/lowest, deep dive |
| 📈 Trend Analysis | Monthly, yearly, seasonal patterns |
| 🔮 Forecasting | 12-month forecast with confidence intervals |
| 🤖 Model Performance | RMSE, R², radar chart comparison |
| 💡 Insights | Automated findings & recommendations |

---

## 🤖 ML Models Trained

- **Linear Regression** – Baseline model
- **Decision Tree Regressor** – Non-linear patterns
- **Random Forest Regressor** – Ensemble, robust to noise
- **Gradient Boosting Regressor** – Typically best performer

Metrics: MAE · MSE · RMSE · R²

---

## 📈 Sample Insights Generated

- State with highest average unemployment
- State with lowest average unemployment
- COVID-19 impact analysis (2020 spike)
- Seasonal unemployment patterns
- Post-pandemic recovery trends

---

## 🔮 Forecast Outputs

| Horizon | Output |
|---|---|
| Next Month | Point forecast |
| Next Quarter | Point forecast |
| Next Year (12 months) | Full series + confidence band |

---

## 🔧 Future Improvements

- [ ] Integrate CMIE / MOSPI live data API
- [ ] Add GDP, CPI, and sector employment co-variates
- [ ] LSTM / Transformer forecasting model
- [ ] State-level individual forecast pages
- [ ] Docker containerization + cloud deployment

---

## 📄 License

This project is for educational and research purposes.
Data is synthetically generated to mirror CMIE unemployment patterns.

---

*Built with Python · Pandas · Scikit-Learn · Plotly · Streamlit*
