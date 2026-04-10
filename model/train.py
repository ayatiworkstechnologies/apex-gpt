"""
Construction Material Estimator - Model Training
=================================================
Model: MultiOutputRegressor wrapping RandomForestRegressor
Targets: cement_bags, sand_cft, bricks, aggregate_cft, steel_kg
Features: area_sqft, floors, building_type, quality, total_area_sqft, city
"""

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/data.csv")
FALLBACK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "estimator_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

NUMERIC_FEATURES = ["area_sqft", "floors", "building_type", "quality", "total_area_sqft"]
CATEGORICAL_FEATURES = ["city"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]

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

    if "city" not in df.columns:
        df["city"] = "chennai"

    df["city"] = df["city"].map(_normalise_city_name)

    X = df[FEATURES]
    y = df[TARGETS]
    return X, y, source_path


def build_pipeline():
    """Build a quantity model that can learn from numeric and city features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("city", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        random_state=42,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", MultiOutputRegressor(rf, n_jobs=1)),
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

    # Disable parallel jobs before saving so API inference runs synchronously and ultra-fast
    pipeline.named_steps["model"].n_jobs = 1
    pipeline.named_steps["model"].estimator.n_jobs = 1

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    meta = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
        "data_source": source_path,
        "cities_seen": sorted(X["city"].unique().tolist()),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
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
