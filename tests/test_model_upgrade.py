import json

from app.predictor import predict
from app.city_rates import get_cost_estimate


def test_model_metadata_uses_ratio_estimator():
    with open("model/model_meta.json", encoding="utf-8") as file:
        meta = json.load(file)

    assert meta["model_type"] == "QuantityRatioRegressor(MultiOutputRegressor(RandomForestRegressor))"
    assert "foundation_factor" in meta["features"]
    assert meta["metrics"]["cement_bags"]["r2"] >= 0.98
    assert meta["metrics"]["steel_kg"]["r2"] >= 0.994


def test_prediction_scales_with_total_area():
    single_floor = predict(
        area=1000,
        unit="sqft",
        floors=1,
        building_type=0,
        quality=1,
        city="chennai",
    )
    two_floor = predict(
        area=1000,
        unit="sqft",
        floors=2,
        building_type=0,
        quality=1,
        city="chennai",
    )

    assert single_floor["materials"]["cement_bags"] > 350
    assert single_floor["materials"]["cement_bags"] < 500
    assert two_floor["materials"]["cement_bags"] > single_floor["materials"]["cement_bags"] * 1.9
    assert two_floor["materials"]["steel_kg"] > single_floor["materials"]["steel_kg"] * 1.9


def test_chennai_aggregate_rate_range_is_available():
    materials = {
        "cement_bags": 100,
        "sand_cft": 100,
        "bricks": 100,
        "aggregate_cft": 100,
        "steel_kg": 100,
    }

    cost = get_cost_estimate(materials, city_key="chennai", total_sqft=1000, quality=1)

    assert cost["rates_used"]["aggregate_per_cft"] == 45
    assert cost["rate_ranges"]["aggregate_per_cft"] == "39-51"
