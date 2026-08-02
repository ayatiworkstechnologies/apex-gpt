"""
Local LLM-backed prompt parser (via Ollama).

Sends the free-text construction prompt to a locally-running Ollama model
and asks it to return structured fields as JSON. No cloud API calls — the
model runs entirely on the machine hosting Ollama.

This is opt-in: `parse_prompt()` in `nlp_parser.py` only calls into this
module when `PROMPT_PARSER=llm` is set, and falls back to the regex parser
on any failure (Ollama not running, model not pulled, malformed JSON, etc.)
so the API keeps working even if the local LLM is unavailable.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:   ollama pull llama3.2
    3. Set env vars:   PROMPT_PARSER=llm
                        OLLAMA_MODEL=llama3.2       (optional, default shown)
                        OLLAMA_URL=http://localhost:11434  (optional, default shown)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))

BUILDING_TYPES = {"residential": 0, "commercial": 1, "industrial": 2}
QUALITY_GRADES = {"economy": 0, "standard": 1, "premium": 2}

SYSTEM_PROMPT = """You extract structured fields from a construction project description.
Respond with ONLY a JSON object (no prose, no markdown fences) with exactly these keys:

{
  "area": <number or null>,            // built-up area if explicitly stated
  "unit": <"sqft" or "sqm" or null>,    // unit for "area", null if area is null
  "floors": <integer or null>,          // number of floors, e.g. "G+2" means 3
  "bhk": <integer or null>,             // bedroom count if mentioned (e.g. "3 BHK" -> 3)
  "building_type": <"residential" or "commercial" or "industrial">,
  "quality": <"economy" or "standard" or "premium">,
  "city": <string or null>,             // city name if mentioned
  "state": <string or null>             // Indian state name if mentioned and no city given
}

Rules:
- If area is not explicitly stated in the text, set "area" and "unit" to null (do not guess a number).
- Default "building_type" to "residential" and "quality" to "standard" if not stated.
- "floors" for "G+N" phrasing is N+1. Plain "N floors" is N. Default 1 if unstated.
- Output must be valid JSON and nothing else."""


class LLMParseError(RuntimeError):
    """Raised when the local LLM is unreachable or returns unusable output."""


def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMParseError(f"Could not reach Ollama at {OLLAMA_URL}: {exc}") from exc

    text = body.get("response")
    if not text:
        raise LLMParseError("Ollama returned an empty response")
    return text


def _coerce_fields(raw: dict[str, Any]) -> dict[str, Any]:
    area = raw.get("area")
    unit = raw.get("unit")
    if area is not None:
        try:
            area = float(area)
        except (TypeError, ValueError):
            area = None
    unit = unit if unit in ("sqft", "sqm") else ("sqft" if area is not None else None)

    floors = raw.get("floors")
    try:
        floors = max(1, min(int(floors), 15)) if floors is not None else 1
    except (TypeError, ValueError):
        floors = 1

    bhk = raw.get("bhk")
    try:
        bhk = int(bhk) if bhk is not None else None
    except (TypeError, ValueError):
        bhk = None

    building_type = BUILDING_TYPES.get(str(raw.get("building_type", "")).lower(), 0)
    quality = QUALITY_GRADES.get(str(raw.get("quality", "")).lower(), 1)

    city = raw.get("city") or None
    state = raw.get("state") or None

    return {
        "area": area,
        "unit": unit,
        "floors": floors,
        "bhk": bhk,
        "building_type": building_type,
        "quality": quality,
        "city": city,
        "state": state,
    }


def parse_prompt_llm(text: str) -> dict:
    """
    Parse a free-text construction prompt using a local Ollama model.

    Returns a dict with the same shape as `nlp_parser._coerce_fields()` output
    (area, unit, floors, bhk, building_type, quality, city, state). Raises
    LLMParseError if Ollama is unreachable or the response isn't usable JSON —
    callers should catch this and fall back to the regex parser.
    """
    raw_text = _call_ollama(text.strip())
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Ollama returned non-JSON output: {raw_text[:200]!r}") from exc

    if not isinstance(raw, dict):
        raise LLMParseError(f"Ollama JSON was not an object: {raw_text[:200]!r}")

    return _coerce_fields(raw)
