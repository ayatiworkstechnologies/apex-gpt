"""Direct inference tests — no server required."""

import pytest

from app.predictor import predict

CASES = [
    ("minimal", dict(area=1000, unit="sqft", floors=1, building_type=0, quality=1)),
    ("commercial_sqm", dict(area=250, unit="sqm", floors=3, building_type=1, quality=2)),
    ("economy_residential", dict(area=600, unit="sqft", floors=1, building_type=0, quality=0)),
    ("industrial", dict(area=5000, unit="sqft", floors=4, building_type=2, quality=1)),
]

MATERIAL_KEYS = ["cement_bags", "sand_cft", "bricks", "aggregate_cft", "steel_kg"]


@pytest.mark.parametrize("desc,payload", CASES, ids=[c[0] for c in CASES])
def test_predict_returns_positive_material_quantities(desc, payload):
    result = predict(**payload)

    assert result["total_area_sqft"] > 0
    for key in MATERIAL_KEYS:
        assert key in result["materials"]
        assert result["materials"][key] >= 0


def test_predict_scales_with_total_area():
    small = predict(area=1000, unit="sqft", floors=1, building_type=0, quality=1)
    large = predict(area=2000, unit="sqft", floors=1, building_type=0, quality=1)

    assert large["total_area_sqft"] > small["total_area_sqft"]
    assert large["materials"]["cement_bags"] > small["materials"]["cement_bags"]


def test_predict_sqm_converted_to_sqft():
    result = predict(area=100, unit="sqm", floors=1, building_type=0, quality=1)

    assert result["input_area_sqft"] == pytest.approx(1076.39, rel=1e-3)
