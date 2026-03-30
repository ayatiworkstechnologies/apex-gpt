"""
Construction Material Estimator - Model Training
=================================================
Model  : MultiOutputRegressor wrapping RandomForestRegressor
Targets: cement_bags, sand_cft, bricks, aggregate_cft, steel_kg
Features: area_sqft, floors, building_type, quality, total_area_sqft
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "../data/training_data.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "estimator_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")

FEATURES = ["area_sqft", "floors", "building_type", "quality", "total_area_sqft"]
TARGETS  = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGETS]

# ── Build Pipeline ─────────────────────────────────────────────────────────────
def build_pipeline():
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  MultiOutputRegressor(rf, n_jobs=-1)),
    ])

# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(pipeline, X_test, y_test):
    y_pred  = pipeline.predict(X_test)
    results = {}
    for i, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        r2  = r2_score(y_test.iloc[:, i], y_pred[:, i])
        results[target] = {"mae": round(mae, 2), "r2": round(r2, 4)}
        print(f"  {target:<22} MAE={mae:>8.1f}   R²={r2:.4f}")
    return results

# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    print("Loading data …")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = build_pipeline()
    print("Training …")
    pipeline.fit(X_train, y_train)

    print("\n── Evaluation on held-out test set ──")
    metrics = evaluate(pipeline, X_test, y_test)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    meta = {
        "features":      FEATURES,
        "targets":       TARGETS,
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "metrics":       metrics,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta  saved → {META_PATH}")

    return pipeline, metrics

# ── Quick CLI test ─────────────────────────────────────────────────────────────
def predict_single(pipeline, area_sqft, floors, building_type=0, quality=1):
    X = pd.DataFrame([{
        "area_sqft":       area_sqft,
        "floors":          floors,
        "building_type":   building_type,
        "quality":         quality,
        "total_area_sqft": area_sqft * floors,
    }])
    pred = pipeline.predict(X)[0]
    return {t: max(0, int(round(v))) for t, v in zip(TARGETS, pred)}


if __name__ == "__main__":
    pipeline, _ = train()
    print("\n── Sample Prediction ──")
    sample = predict_single(pipeline, area_sqft=1200, floors=2,
                            building_type=0, quality=1)
    for k, v in sample.items():
        print(f"  {k:<22} {v:>8,}")