"""
Model loader & prediction service.
Loads the trained sklearn pipeline once at startup (LRU-cached).
"""

import os
import json
import joblib
import pandas as pd
from functools import lru_cache

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/estimator_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "../model/model_meta.json")

FEATURES = ["area_sqft", "floors", "building_type", "quality", "total_area_sqft"]
TARGETS  = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]

BUILDING_LABELS = {0: "Residential", 1: "Commercial", 2: "Industrial"}
QUALITY_LABELS  = {0: "Economy",     1: "Standard",   2: "Premium"}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained sklearn pipeline."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_meta() -> dict:
    """Load model metadata — R² scores, feature list, sample counts."""
    with open(META_PATH) as f:
        return json.load(f)


def sqm_to_sqft(sqm: float) -> float:
    return sqm * 10.7639


def predict(
    area: float,
    unit: str,
    floors: int,
    building_type: int,
    quality: int,
) -> dict:
    """
    Run inference on a single request.

    Parameters
    ----------
    area          : Raw area value entered by user.
    unit          : 'sqft' or 'sqm'  (auto-converts sqm → sqft).
    floors        : Number of floors.
    building_type : 0=Residential, 1=Commercial, 2=Industrial.
    quality       : 0=Economy, 1=Standard, 2=Premium.

    Returns
    -------
    dict with material quantities + metadata.
    """
    area_sqft  = sqm_to_sqft(area) if unit == "sqm" else area
    total_area = area_sqft * floors

    X = pd.DataFrame([{
        "area_sqft":       area_sqft,
        "floors":          floors,
        "building_type":   building_type,
        "quality":         quality,
        "total_area_sqft": total_area,
    }])

    preds     = load_model().predict(X)[0]
    materials = {t: max(0, int(round(v))) for t, v in zip(TARGETS, preds)}
    meta      = load_meta()

    return {
        "input_area_sqft": round(area_sqft, 2),
        "total_area_sqft": round(total_area, 2),
        "building_type":   BUILDING_LABELS[building_type],
        "quality":         QUALITY_LABELS[quality],
        "materials":       materials,
        "model_r2_scores": {t: meta["metrics"][t]["r2"] for t in TARGETS},
    }