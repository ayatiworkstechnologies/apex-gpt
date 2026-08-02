"""Tests for NLP prompt parsing + prediction, simulating /estimate-from-prompt internals."""

import pytest

from app.nlp_parser import parse_prompt
from app.predictor import predict

BUILDING_LABELS = {0: "Residential", 1: "Commercial", 2: "Industrial"}
QUALITY_LABELS = {0: "Economy", 1: "Standard", 2: "Premium"}

TEST_PROMPTS = [
    "3 BHK 3 floor residential house",
    "2BHK house 1200 sqft G+1 standard quality",
    "commercial building 5 floors 250 sqm premium",
    "small economy house 600 sqft",
    "4 BHK luxury villa G+2",
    "industrial warehouse 5000 sqft 2 floors",
    "1 BHK apartment",
    "build me a 3 bhk home",
    "5 BHK premium residential G+3",
    "office building 10000 sqft 4 floors standard",
]


def _estimate_from_prompt(prompt: str):
    parsed = parse_prompt(prompt)
    result = predict(
        area=parsed["area"],
        unit=parsed["unit"],
        floors=parsed["floors"],
        building_type=parsed["building_type"],
        quality=parsed["quality"],
    )
    return parsed, result


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_prompt_parses_and_predicts_without_error(prompt):
    parsed, result = _estimate_from_prompt(prompt)

    assert parsed["building_type"] in BUILDING_LABELS
    assert parsed["quality"] in QUALITY_LABELS
    assert parsed["area"] > 0
    assert parsed["floors"] >= 1
    assert result["total_area_sqft"] > 0
    assert all(v >= 0 for v in result["materials"].values())


def test_bhk_prompt_infers_bhk_count():
    parsed, _ = _estimate_from_prompt("3 BHK 3 floor residential house")

    assert parsed["bhk"] == 3
    assert parsed["building_type"] == 0


def test_g_plus_n_prompt_infers_floor_count():
    parsed, _ = _estimate_from_prompt("4 BHK luxury villa G+2")

    assert parsed["floors"] == 3


def test_multi_digit_bhk_is_parsed_correctly():
    parsed, _ = _estimate_from_prompt("10 BHK luxury villa in Mumbai")

    assert parsed["bhk"] == 10
