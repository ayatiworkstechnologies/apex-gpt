"""Tests for the local Ollama-backed prompt parser and its fallback behavior."""

import json

import pytest

import app.llm_parser as llm_parser
import app.nlp_parser as nlp_parser


def _ollama_json(**overrides):
    base = {
        "area": None, "unit": None, "floors": 1, "bhk": None,
        "building_type": "residential", "quality": "standard",
        "city": None, "state": None,
    }
    base.update(overrides)
    return json.dumps(base)


def test_parse_prompt_llm_coerces_fields(monkeypatch):
    monkeypatch.setattr(
        llm_parser, "_call_ollama",
        lambda prompt: _ollama_json(area=2000, unit="sqft", floors=3, bhk=None,
                                     building_type="commercial", quality="premium", city="Mumbai"),
    )

    fields = llm_parser.parse_prompt_llm("premium commercial building 2000 sqft 3 floors in Mumbai")

    assert fields["area"] == 2000.0
    assert fields["unit"] == "sqft"
    assert fields["floors"] == 3
    assert fields["building_type"] == 1  # commercial
    assert fields["quality"] == 2  # premium
    assert fields["city"] == "Mumbai"


def test_parse_prompt_llm_raises_on_non_json(monkeypatch):
    monkeypatch.setattr(llm_parser, "_call_ollama", lambda prompt: "not valid json")

    with pytest.raises(llm_parser.LLMParseError):
        llm_parser.parse_prompt_llm("some prompt")


def test_parse_prompt_llm_clamps_out_of_range_floors(monkeypatch):
    monkeypatch.setattr(llm_parser, "_call_ollama", lambda prompt: _ollama_json(floors=99))

    fields = llm_parser.parse_prompt_llm("some prompt")

    assert fields["floors"] == 15


def test_nlp_parser_uses_llm_when_enabled(monkeypatch):
    monkeypatch.setattr(nlp_parser, "PROMPT_PARSER", "llm")
    monkeypatch.setattr(
        llm_parser, "_call_ollama",
        lambda prompt: _ollama_json(area=1200, unit="sqft", floors=2, bhk=3,
                                     building_type="residential", quality="standard", city="Chennai"),
    )

    result = nlp_parser.parse_prompt("3 BHK 1200 sqft house in Chennai")

    assert result["area"] == 1200.0
    assert result["bhk"] == 3
    assert result["city"] == "Chennai"
    assert "Parsed via local LLM" in result["parsed_notes"][0]


def test_nlp_parser_falls_back_to_regex_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(nlp_parser, "PROMPT_PARSER", "llm")
    monkeypatch.setattr(
        llm_parser, "_call_ollama",
        lambda prompt: (_ for _ in ()).throw(llm_parser.LLMParseError("Ollama not running")),
    )

    result = nlp_parser.parse_prompt("3 BHK house in Chennai")

    assert result["bhk"] == 3
    assert "Parsed via local LLM" not in " ".join(result["parsed_notes"])


def test_nlp_parser_propagates_validation_error_from_llm(monkeypatch):
    monkeypatch.setattr(nlp_parser, "PROMPT_PARSER", "llm")
    monkeypatch.setattr(llm_parser, "_call_ollama", lambda prompt: _ollama_json())

    with pytest.raises(ValueError, match="Area is required"):
        nlp_parser.parse_prompt("vague prompt with no area or bhk")
