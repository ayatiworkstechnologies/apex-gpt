from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_live_refresh_endpoint_runs_price_update_and_tuning(monkeypatch):
    import app.live_refresh as live_refresh
    import app.main as main

    monkeypatch.setattr(main, "REFRESH_API_KEY", "test-key")

    monkeypatch.setattr(
        live_refresh,
        "update_prices",
        lambda dry_run, only_verified: {
            "updated": [{"city": "chennai"}],
            "skipped": [],
            "failed": [],
            "dry_run": dry_run,
        },
    )
    monkeypatch.setattr(live_refresh, "reload_city_db", lambda: {"chennai": {"cement": 420}})
    monkeypatch.setattr(
        live_refresh,
        "run_tuning",
        lambda save_best, version: {
            "version": version or "2026.05.20",
            "selected_candidate": "balanced",
            "score": {"avg_r2": 0.99, "avg_mae": 100.0},
            "metrics": {"cement_bags": {"r2": 0.99, "mae": 100.0}},
        },
    )
    monkeypatch.setattr(live_refresh, "reload_runtime_artifacts", lambda: {"model_loaded": True, "meta_loaded": True})
    monkeypatch.setattr(
        live_refresh,
        "get_model_catalog",
        lambda: {
            "version": "2026.05.20",
            "trained_at": None,
            "auto_tuned": True,
            "model_type": "QuantityRatioRegressor(MultiOutputRegressor(RandomForestRegressor))",
            "active_candidate": "balanced",
            "model_params": {},
            "features": ["area_sqft", "city"],
            "numeric_features": ["area_sqft"],
            "categorical_features": ["city"],
            "targets": ["cement_bags"],
            "train_samples": 10,
            "test_samples": 2,
            "data_source": "data/data.csv",
            "cities_seen": ["chennai"],
            "score": {"avg_r2": 0.99, "avg_mae": 100.0},
            "metrics": {"cement_bags": {"r2": 0.99, "mae": 100.0}},
            "candidates": [],
        },
    )

    response = client.post(
        "/api/model/refresh-live",
        json={"only_verified": True, "version": "2026.05.20"},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "completed"
    assert payload["summary"]["price_update"]["updated"][0]["city"] == "chennai"
    assert payload["summary"]["model_tuning"]["selected_candidate"] == "balanced"
    assert payload["summary"]["runtime_reload"]["model_loaded"] is True


def test_live_refresh_endpoint_rejects_missing_api_key(monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "REFRESH_API_KEY", "test-key")

    response = client.post("/api/model/refresh-live", json={"only_verified": True})

    assert response.status_code == 401


def test_live_refresh_status_endpoint_returns_last_status():
    response = client.get("/api/model/refresh-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] in {"idle", "running", "completed", "failed"}
