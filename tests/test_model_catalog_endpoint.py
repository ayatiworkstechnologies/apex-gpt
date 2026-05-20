from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_model_catalog_endpoint_returns_structured_model_summary():
    response = client.get("/api/model/catalog")

    assert response.status_code == 200

    payload = response.json()
    assert payload["model_type"] == "QuantityRatioRegressor(MultiOutputRegressor(RandomForestRegressor))"
    assert "foundation_factor" in payload["features"]
    assert payload["features"][-1] == "city"
    assert payload["metrics"]["cement_bags"]["r2"] >= 0.98
    assert payload["targets"] == [
        "cement_bags",
        "sand_cft",
        "bricks",
        "aggregate_cft",
        "steel_kg",
    ]
