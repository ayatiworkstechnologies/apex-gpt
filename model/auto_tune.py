"""
Daily auto-tuning for the Apex construction estimator.

Runs a small, deterministic hyperparameter search, keeps the best candidate,
and writes a dated version record. Intended for a daily scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import joblib
from sklearn.model_selection import train_test_split

from model.train import (
    FEATURES,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    TARGETS,
    build_pipeline,
    evaluate,
    load_data,
    score_metrics,
)

VERSION_DIR = os.path.join(MODEL_DIR, "versions")
TUNING_LOG_PATH = os.path.join(MODEL_DIR, "tuning_history.jsonl")

DEFAULT_CANDIDATES = [
    {"name": "balanced", "n_estimators": 160, "max_depth": 9, "min_samples_leaf": 8},
    {"name": "recall_plus", "n_estimators": 200, "max_depth": 10, "min_samples_leaf": 6},
    {"name": "stable_smooth", "n_estimators": 180, "max_depth": 8, "min_samples_leaf": 10},
]


def _daily_version() -> str:
    return datetime.now().strftime("%Y.%m.%d")


def _candidate_score_key(result: dict) -> tuple:
    score = result["score"]
    # Prefer higher R2 first, then lower MAE.
    return (score["avg_r2"], -score["avg_mae"])


def _single_thread_for_api(pipeline):
    ratio_model = pipeline.named_steps["model"]
    ratio_model.estimator_.n_jobs = 1
    for estimator in ratio_model.estimator_.estimators_:
        estimator.n_jobs = 1


def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _append_tuning_log(record: dict):
    with open(TUNING_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_tuning(save_best: bool = True, version: str | None = None) -> dict:
    os.makedirs(VERSION_DIR, exist_ok=True)
    version = version or _daily_version()

    print("Loading data for daily tuning...")
    X, y, source_path = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_pipeline = None
    for params in DEFAULT_CANDIDATES:
        name = params["name"]
        model_params = {key: value for key, value in params.items() if key != "name"}
        print(f"\n-- Candidate: {name} {model_params} --")
        pipeline = build_pipeline(**model_params)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        score = score_metrics(metrics)
        result = {
            "candidate": name,
            "model_params": model_params,
            "metrics": metrics,
            "score": score,
        }
        print(f"  avg_mae={score['avg_mae']} avg_r2={score['avg_r2']}")
        results.append(result)
        if result == max(results, key=_candidate_score_key):
            best_pipeline = pipeline

    best = max(results, key=_candidate_score_key)
    timestamp = datetime.now().isoformat(timespec="seconds")
    meta = {
        "version": version,
        "trained_at": timestamp,
        "model_type": "QuantityRatioRegressor(MultiOutputRegressor(RandomForestRegressor))",
        "auto_tuned": True,
        "selected_candidate": best["candidate"],
        "model_params": best["model_params"],
        "features": FEATURES,
        "targets": TARGETS,
        "data_source": source_path,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": best["metrics"],
        "score": best["score"],
        "candidates": results,
    }

    if save_best:
        _single_thread_for_api(best_pipeline)
        joblib.dump(best_pipeline, MODEL_PATH)
        _write_json(META_PATH, meta)

        version_model_path = os.path.join(VERSION_DIR, f"estimator_model_{version}.pkl")
        version_meta_path = os.path.join(VERSION_DIR, f"model_meta_{version}.json")
        joblib.dump(best_pipeline, version_model_path)
        _write_json(version_meta_path, meta)

    _append_tuning_log({
        "version": version,
        "trained_at": timestamp,
        "selected_candidate": best["candidate"],
        "score": best["score"],
        "saved": save_best,
    })

    print(f"\nSelected: {best['candidate']} | avg_r2={best['score']['avg_r2']} avg_mae={best['score']['avg_mae']}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Daily model auto-tuning")
    parser.add_argument("--no-save", action="store_true", help="Run tuning without replacing the live model")
    parser.add_argument("--version", help="Version label, defaults to YYYY.MM.DD")
    args = parser.parse_args()

    run_tuning(save_best=not args.no_save, version=args.version)


if __name__ == "__main__":
    main()
