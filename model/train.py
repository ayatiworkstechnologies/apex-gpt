"""
Construction Material Estimator - Model Training
=================================================
Model: QuantityRatioRegressor wrapping MultiOutputRegressor(RandomForestRegressor)
Targets: cement_bags, sand_cft, bricks, aggregate_cft, steel_kg
Features: area_sqft, floors, building_type, quality, total_area_sqft,
          foundation_factor, city
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.quantity_model import QuantityRatioRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/data.csv")
FALLBACK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "estimator_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

NUMERIC_FEATURES = [
    "area_sqft",
    "floors",
    "building_type",
    "quality",
    "total_area_sqft",
    "foundation_factor",
]
CATEGORICAL_FEATURES = ["city"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]
TOTAL_AREA_FEATURE_INDEX = NUMERIC_FEATURES.index("total_area_sqft")

os.makedirs(MODEL_DIR, exist_ok=True)


def _normalise_city_name(value: str) -> str:
    text = str(value or "chennai").strip().lower()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text or "chennai"


def load_data(path: str = DATA_PATH):
    source_path = path if os.path.exists(path) else FALLBACK_DATA_PATH
    df = pd.read_csv(source_path)

    missing_targets = [col for col in TARGETS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Training CSV is missing target columns: {missing_targets}")

    for column in ["area_sqft", "floors", "building_type", "quality"]:
        if column not in df.columns:
            raise ValueError(f"Training CSV is missing required feature column: {column}")

    if "total_area_sqft" not in df.columns:
        df["total_area_sqft"] = df["area_sqft"] * df["floors"]

    df["foundation_factor"] = 1.0 + ((df["floors"] - 1) * 0.018).clip(0, 0.20)

    if "city" not in df.columns:
        df["city"] = "chennai"

    df["city"] = df["city"].map(_normalise_city_name)

    X = df[FEATURES]
    y = df[TARGETS]
    return X, y, source_path


def build_pipeline(
    n_estimators: int = 160,
    max_depth: int = 9,
    min_samples_leaf: int = 8,
    random_state: int = 42,
):
    """Build a quantity model that can learn from numeric and city features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "city",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            QuantityRatioRegressor(
                estimator=MultiOutputRegressor(rf, n_jobs=1),
                total_area_index=TOTAL_AREA_FEATURE_INDEX,
            ),
        ),
    ])


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    results = {}
    has_enough_test_rows = len(X_test) > 1
    for idx, target in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, idx], y_pred[:, idx])
        r2 = r2_score(y_test.iloc[:, idx], y_pred[:, idx]) if has_enough_test_rows else 0.0
        results[target] = {"mae": round(mae, 2), "r2": round(r2, 4)}
        print(f"  {target:<22} MAE={mae:>8.1f}   R2={r2:.4f}")
    return results


def score_metrics(metrics: dict) -> dict:
    mae_values = [item["mae"] for item in metrics.values()]
    r2_values = [item["r2"] for item in metrics.values()]
    return {
        "avg_mae": round(sum(mae_values) / len(mae_values), 2),
        "avg_r2": round(sum(r2_values) / len(r2_values), 4),
    }


def train():
    print("Loading data...")
    X, y, source_path = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Source CSV: {source_path}")

    print("\nBuilding pipeline...")
    pipeline = build_pipeline()

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("\n-- Evaluation on held-out test set --")
    metrics = evaluate(pipeline, X_test, y_test)

    # Keep the persisted model single-threaded for predictable API inference.
    ratio_model = pipeline.named_steps["model"]
    ratio_model.estimator_.n_jobs = 1
    for estimator in ratio_model.estimator_.estimators_:
        estimator.n_jobs = 1

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    meta = {
        "model_type": "QuantityRatioRegressor(MultiOutputRegressor(RandomForestRegressor))",
        "model_params": {
            "n_estimators": 160,
            "max_depth": 9,
            "min_samples_leaf": 8,
            "random_state": 42,
        },
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "total_area_feature_index": TOTAL_AREA_FEATURE_INDEX,
        "targets": TARGETS,
        "data_source": source_path,
        "cities_seen": sorted(X["city"].unique().tolist()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "score": score_metrics(metrics),
    }
    with open(META_PATH, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(f"Meta saved  -> {META_PATH}")

    return pipeline, metrics


def predict_single(
    pipeline,
    area_sqft: float,
    floors: int,
    building_type: int = 0,
    quality: int = 1,
    city: str = "chennai",
):
    X = pd.DataFrame([{
        "area_sqft": area_sqft,
        "floors": floors,
        "building_type": building_type,
        "quality": quality,
        "total_area_sqft": area_sqft * floors,
        "foundation_factor": 1.0 + min(max((floors - 1) * 0.018, 0), 0.20),
        "city": _normalise_city_name(city),
    }])
    pred = pipeline.predict(X)[0]
    return {target: max(0, int(round(value))) for target, value in zip(TARGETS, pred)}


if __name__ == "__main__":
    model, metrics = train()

    print("\n-- Sample Prediction --")
    sample = predict_single(
        model,
        area_sqft=1200,
        floors=2,
        building_type=0,
        quality=1,
        city="chennai",
    )
    for key, value in sample.items():
        print(f"  {key:<22} {value:>8,}")
