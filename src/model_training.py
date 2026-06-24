"""
model_training.py
=================
Trains, evaluates, and compares multiple regression models.
Selects the best model based on RMSE and saves it to disk.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    """
    Return a dictionary of model name → sklearn estimator instances.

    Returns
    -------
    dict
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05,
            max_depth=5, random_state=42
        ),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
        MAE, MSE, RMSE, R2 scores.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all models, evaluate on test set, and return results.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series

    Returns
    -------
    tuple
        (results_dict, best_model_name, best_model_object, trained_models_dict)
    """
    models = get_models()
    results = {}
    trained_models = {}

    print("\n[TRAINING] Training and evaluating models...")
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results[name] = metrics
        trained_models[name] = model

        print(f"  {name:<25} | MAE: {metrics['MAE']:.4f} | "
              f"RMSE: {metrics['RMSE']:.4f} | R²: {metrics['R2']:.4f}")

    # Select best model (lowest RMSE)
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = trained_models[best_name]

    print(f"\n  ✅ Best Model: {best_name} (RMSE={results[best_name]['RMSE']:.4f})")
    return results, best_name, best_model, trained_models


def save_best_model(model, model_name: str, feature_cols: list) -> str:
    """
    Serialize the best model and its feature list to disk.

    Parameters
    ----------
    model : sklearn estimator
    model_name : str
    feature_cols : list of str

    Returns
    -------
    str
        Path to saved model file.
    """
    safe_name = model_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"best_model_{safe_name}.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump({"model_name": model_name, "feature_cols": feature_cols}, f)

    print(f"[SAVE] Best model saved: {model_path}")
    return model_path


def save_comparison_report(results: dict) -> str:
    """
    Save a CSV model comparison report.

    Parameters
    ----------
    results : dict
        {model_name: {MAE, MSE, RMSE, R2}, ...}

    Returns
    -------
    str
        Path to the comparison CSV.
    """
    report_df = pd.DataFrame(results).T.reset_index()
    report_df.rename(columns={"index": "Model"}, inplace=True)
    report_df.sort_values("RMSE", inplace=True)

    path = os.path.join(MODELS_DIR, "model_comparison.csv")
    report_df.to_csv(path, index=False)
    print(f"[SAVE] Model comparison report saved: {path}")
    return path
