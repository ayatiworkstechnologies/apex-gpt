"""
Model loader & prediction service
Loads the trained pipeline once at startup and exposes predict().
"""

import os
import json
import joblib
import pandas as pd
from functools import lru_cache

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/estimator_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "../model/model_meta.json")

FEATURES = [
    "area_sqft",
    "floors",
    "building_type",
    "quality",
    "total_area_sqft",
    "foundation_factor",
    "city",
]
TARGETS  = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]

BUILDING_LABELS = {0: "Residential", 1: "Commercial", 2: "Industrial"}
QUALITY_LABELS  = {0: "Economy",     1: "Standard",   2: "Premium"}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained sklearn pipeline."""
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


@lru_cache(maxsize=1)
def load_meta() -> dict:
    """Load model metadata (R² scores, feature list etc.)."""
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def reload_runtime_artifacts() -> dict:
    """Clear caches and reload the live model + metadata from disk."""
    load_model.cache_clear()
    load_meta.cache_clear()
    pipeline = load_model()
    meta = load_meta()
    return {"model_loaded": pipeline is not None, "meta_loaded": bool(meta)}


def _candidate_rank_key(candidate: dict) -> tuple:
    score = candidate.get("score", {})
    return (
        float(score.get("avg_r2", 0.0)),
        -float(score.get("avg_mae", 0.0)),
    )


def get_model_catalog() -> dict:
    """Return a structured summary of the active AI model and tuned candidates."""
    meta = load_meta()
    candidates = sorted(
        meta.get("candidates", []),
        key=_candidate_rank_key,
        reverse=True,
    )
    active_candidate = meta.get("selected_candidate")
    if not active_candidate and candidates:
        active_candidate = candidates[0].get("candidate")

    return {
        "version": meta.get("version", "live"),
        "trained_at": meta.get("trained_at"),
        "auto_tuned": bool(meta.get("auto_tuned", False)),
        "model_type": meta.get("model_type", "unknown"),
        "active_candidate": active_candidate,
        "model_params": meta.get("model_params", {}),
        "features": meta.get("features", FEATURES),
        "numeric_features": meta.get("numeric_features", []),
        "categorical_features": meta.get("categorical_features", []),
        "targets": meta.get("targets", TARGETS),
        "train_samples": int(meta.get("train_samples", 0)),
        "test_samples": int(meta.get("test_samples", 0)),
        "data_source": meta.get("data_source", ""),
        "cities_seen": meta.get("cities_seen", []),
        "score": meta.get("score"),
        "metrics": meta.get("metrics", {}),
        "candidates": candidates,
    }


def sqm_to_sqft(sqm: float) -> float:
    return sqm * 10.7639


def normalize_city_name(city: str | None) -> str:
    text = str(city or "chennai").strip().lower()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text or "chennai"


def foundation_factor(floors: int) -> float:
    return 1.0 + min(max((floors - 1) * 0.018, 0), 0.20)


def predict(
    area: float,
    unit: str,
    floors: int,
    building_type: int,
    quality: int,
    city: str | None = None,
) -> dict:
    """
    Run inference on a single request.

    Parameters
    ----------
    area          : Raw area entered by user.
    unit          : 'sqft' or 'sqm'.
    floors        : Number of floors.
    building_type : 0/1/2  (residential / commercial / industrial).
    quality       : 0/1/2  (economy / standard / premium).

    Returns
    -------
    dict with prediction and metadata.
    """
    area_sqft = sqm_to_sqft(area) if unit == "sqm" else area
    total_area = area_sqft * floors

    X = pd.DataFrame([{
        "area_sqft":       area_sqft,
        "floors":          floors,
        "building_type":   building_type,
        "quality":         quality,
        "total_area_sqft": total_area,
        "foundation_factor": foundation_factor(floors),
        "city":            normalize_city_name(city),
    }])

    pipeline = load_model()
    preds    = pipeline.predict(X)[0]

    materials = {
        t: max(0, int(round(v)))
        for t, v in zip(TARGETS, preds)
    }

    meta = load_meta()

    return {
        "input_area_sqft": round(area_sqft, 2),
        "total_area_sqft": round(total_area, 2),
        "building_type":   BUILDING_LABELS[building_type],
        "quality":         QUALITY_LABELS[quality],
        "materials":       materials,
        "model_r2_scores": {
            t: meta["metrics"][t]["r2"] for t in TARGETS
        },
    }
