"""
prediction.py
=============
Load the best trained model and generate predictions on new or test data.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")
PREDS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions")
os.makedirs(PREDS_DIR, exist_ok=True)


def load_best_model():
    """
    Load the saved best model and its metadata from disk.

    Returns
    -------
    tuple
        (model, model_name, feature_cols) or (None, None, None) if not found.
    """
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        print("[WARN] No model metadata found. Train models first.")
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    feature_cols = meta["feature_cols"]
    safe_name = model_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"best_model_{safe_name}.pkl")

    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found: {model_path}")
        return None, None, None

    model = joblib.load(model_path)
    print(f"[LOAD] Loaded model: {model_name}")
    return model, model_name, feature_cols


def predict(X: pd.DataFrame) -> np.ndarray:
    """
    Run inference using the saved best model.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with the same columns used during training.

    Returns
    -------
    np.ndarray
        Predicted unemployment rates.
    """
    model, model_name, feature_cols = load_best_model()
    if model is None:
        raise RuntimeError("No trained model available. Run main.py first.")

    # Align columns
    X_input = X[[c for c in feature_cols if c in X.columns]]
    predictions = model.predict(X_input)
    return predictions


def save_predictions(y_true: pd.Series, y_pred: np.ndarray,
                     dates: pd.Series = None) -> str:
    """
    Save test set predictions alongside actual values.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray
    dates : pd.Series, optional

    Returns
    -------
    str
        Path to saved predictions CSV.
    """
    pred_df = pd.DataFrame({
        "Actual": y_true.values,
        "Predicted": y_pred.round(2),
        "Error": (y_true.values - y_pred).round(3),
    })
    if dates is not None:
        pred_df.insert(0, "Date", dates.values)

    path = os.path.join(PREDS_DIR, "model_predictions.csv")
    pred_df.to_csv(path, index=False)
    print(f"[SAVE] Predictions saved: {path}")
    return path
