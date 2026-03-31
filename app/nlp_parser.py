"""
Advanced NLP Parser v2
======================
Deep understanding of construction prompts.
Extracts: BHK, floors, area+unit, building type, quality, city, state,
          plot size, budget hints, construction purpose.

Supports:
  "3 BHK 3 floor residential house in Chennai"
  "build 2bhk house 1200 sqft at Coimbatore standard"
  "premium villa G+2 in Bangalore 2400 sqft"
  "commercial office 5 floors 250 sqm Mumbai premium"
  "factory shed 5000 sqft 2 floors Pune"
  "1200 sq ft house chennai 2 floors economy"
"""

import re
from typing import Optional
from app.city_rates import resolve_city, CITY_DB, CITY_ALIASES, STATE_ALIASES

# ── BHK area inference table ───────────────────────────────────────────────────
BHK_AREA_MAP = {1: 500, 2: 850, 3: 1200, 4: 1800, 5: 2400, 6: 3200}

# ── Building type ──────────────────────────────────────────────────────────────
BT_KEYWORDS = {
    2: ["industrial", "factory", "plant", "warehouse", "shed",
        "manufacturing", "workshop", "godown"],
    1: ["commercial", "office", "shop", "mall", "showroom", "hotel",
        "restaurant", "retail", "clinic", "hospital", "school", "college"],
    0: ["residential", "house", "home", "villa", "bungalow", "flat",
        "apartment", "bhk", "duplex", "residence", "row house", "independent"],
}

# ── Quality ────────────────────────────────────────────────────────────────────
QUALITY_KEYWORDS = {
    0: ["economy", "basic", "budget", "low cost", "low-cost", "simple",
        "ordinary", "affordable", "cheap", "minimal"],
    2: ["premium", "luxury", "high end", "high-end", "deluxe", "superior",
        "top class", "best", "executive", "ultra", "royal"],
    1: ["standard", "normal", "regular", "moderate", "average", "typical",
        "middle", "mid"],
}

# ── Floor words ────────────────────────────────────────────────────────────────
FLOOR_WORDS = {
    "one":1,"single":1,"ground":1,
    "two":2,"double":2,
    "three":3,"triple":3,
    "four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
}

# ── All known city names (for extraction from prompt) ─────────────────────────
ALL_CITIES = set(CITY_DB.keys()) | set(CITY_ALIASES.keys())


def _extract_bhk(lower: str, notes: list) -> Optional[int]:
    m = re.search(r'(\d)\s*bhk', lower)
    if m:
        bhk = int(m.group(1))
        notes.append(f"BHK detected: {bhk} BHK")
        return bhk
    return None


def _extract_floors(lower: str, notes: list) -> int:
    # G+N
    m = re.search(r'g\s*\+\s*(\d+)', lower)
    if m:
        f = int(m.group(1)) + 1
        notes.append(f"Floors: G+{m.group(1)} = {f} floors")
        return f
    # N floor(s)
    m = re.search(r'(\d+)\s*-?\s*floors?', lower)
    if m:
        f = min(int(m.group(1)), 15)
        notes.append(f"Floors: {f} (explicit)")
        return f
    # storey/story
    m = re.search(r'(\d+)\s*-?\s*stor(?:ey|y)', lower)
    if m:
        f = min(int(m.group(1)), 15)
        notes.append(f"Floors: {f} (storey)")
        return f
    # word form
    for word, val in FLOOR_WORDS.items():
        if re.search(rf'\b{word}\s+(?:floors?|storey)\b', lower):
            notes.append(f"Floors: {val} (word '{word}')")
            return val
    return 1


def _extract_area(lower: str, notes: list):
    """Returns (area, unit) or (None, None)."""
    # explicit with unit
    m = re.search(
        r'(\d[\d,]*\.?\d*)\s*'
        r'(sq\.?\s?ft|sqft|sq\.?\s?m(?:eter)?|sqm|square\s?f(?:ee)?t|square\s?m(?:eter)?)',
        lower
    )
    if m:
        area = float(m.group(1).replace(",", ""))
        raw_unit = m.group(2).replace(" ", "").replace(".", "").lower()
        unit = "sqm" if ("sqm" in raw_unit or "squaremeter" in raw_unit or "sqmeter" in raw_unit) else "sqft"
        notes.append(f"Area: {area} {unit} (explicit)")
        return area, unit

    # cents (South Indian land unit: 1 cent = 435.6 sqft)
    m = re.search(r'(\d+\.?\d*)\s*cent', lower)
    if m:
        cents = float(m.group(1))
        area = round(cents * 435.6, 1)
        notes.append(f"Area: {cents} cents = {area} sqft (converted)")
        return area, "sqft"

    # ground (Tamil unit: 1 ground = 2400 sqft)
    m = re.search(r'(\d+\.?\d*)\s*grounds?', lower)
    if m:
        grounds = float(m.group(1))
        area = round(grounds * 2400, 1)
        notes.append(f"Area: {grounds} grounds = {area} sqft (converted)")
        return area, "sqft"

    return None, None


def _extract_city(lower: str, raw: str, notes: list) -> Optional[str]:
    """Extract city name from prompt using 'in', 'at', 'near' prepositions or direct match."""
    # after preposition
    prep_m = re.search(r'(?:in|at|near|@)\s+([a-z][a-z\s]{2,20}?)(?:\s|,|$|\.|/)', lower)
    if prep_m:
        candidate = prep_m.group(1).strip().rstrip(",.")
        # multi-word city check
        for length in [3, 2, 1]:
            words = candidate.split()
            if len(words) >= length:
                test = " ".join(words[:length])
                if test in ALL_CITIES or test in CITY_ALIASES:
                    notes.append(f"City: {test.title()} (after preposition)")
                    return test
    # scan all tokens for city match
    tokens = lower.split()
    for i in range(len(tokens)-1, -1, -1):
        # try 2-word combo first
        if i < len(tokens) - 1:
            two = f"{tokens[i]} {tokens[i+1]}"
            if two in ALL_CITIES or two in CITY_ALIASES:
                notes.append(f"City: {two.title()} (token match)")
                return two
        single = tokens[i]
        if single in ALL_CITIES or single in CITY_ALIASES:
            notes.append(f"City: {single.title()} (token match)")
            return single
    return None


def _extract_state(lower: str, notes: list) -> Optional[str]:
    for alias, state in STATE_ALIASES.items():
        if re.search(rf'\b{re.escape(alias)}\b', lower):
            notes.append(f"State: {state}")
            return state
    return None


def _extract_building_type(lower: str, notes: list) -> int:
    for bt in [2, 1, 0]:
        for kw in BT_KEYWORDS[bt]:
            if re.search(rf'\b{re.escape(kw)}\b', lower):
                label = ["Residential", "Commercial", "Industrial"][bt]
                notes.append(f"Building type: {label} ('{kw}')")
                return bt
    notes.append("Building type: Residential (default)")
    return 0


def _extract_quality(lower: str, notes: list) -> int:
    for q in [2, 0, 1]:  # premium first, then economy, then standard
        for kw in QUALITY_KEYWORDS[q]:
            if kw in lower:
                label = ["Economy", "Standard", "Premium"][q]
                notes.append(f"Quality: {label} ('{kw}')")
                return q
    notes.append("Quality: Standard (default)")
    return 1


def parse_prompt(text: str) -> dict:
    """
    Deep parse of free-form construction prompt.

    Mandatory: area with unit (sqft/sqm/cents/grounds)
    Optional : BHK, floors, city, state, building type, quality

    Returns full structured dict.
    """
    raw   = text.strip()
    lower = raw.lower()
    notes = []

    bhk          = _extract_bhk(lower, notes)
    floors       = _extract_floors(lower, notes)
    area, unit   = _extract_area(lower, notes)
    city_raw     = _extract_city(lower, raw, notes)
    state_raw    = _extract_state(lower, notes) if not city_raw else None
    building_type = _extract_building_type(lower, notes)
    quality      = _extract_quality(lower, notes)

    # Area fallback from BHK
    if area is None and bhk is not None:
        area = float(BHK_AREA_MAP.get(bhk, 1000))
        unit = "sqft"
        notes.append(f"Area inferred: {area} sqft from {bhk} BHK")
    elif area is None:
        raise ValueError(
            "Area is required. Please mention area with unit, e.g. '1200 sqft', '250 sqm', '3 cents'."
        )

    # Resolve city (use state fallback if needed)
    city_input = city_raw or state_raw
    city_data  = resolve_city(city_input)

    return {
        "area":          area,
        "unit":          unit,
        "floors":        floors,
        "building_type": building_type,
        "quality":       quality,
        "bhk":           bhk,
        "city":          city_data["city"],
        "state":         city_data["state"],
        "city_data":     city_data,
        "raw_prompt":    raw,
        "parsed_notes":  notes,
    }


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "3 BHK 3 floor residential house in Chennai",
        "2BHK house 1200 sqft G+1 Coimbatore standard",
        "premium villa G+2 in Bangalore 2400 sqft",
        "commercial office 5 floors 250 sqm Mumbai premium",
        "factory shed 5000 sqft 2 floors Pune",
        "1200 sqft house chennai 2 floors economy",
        "4 BHK luxury villa G+3 Hyderabad",
        "small house 600 sqft Delhi budget",
        "3 bhk home 1500 sqft noida standard",
        "industrial warehouse 8000 sqft 3 floors gujarat",
        "3 cents house coimbatore",
        "2 grounds residential house chennai g+1 premium",
    ]
    for t in tests:
        try:
            r = parse_prompt(t)
            print(f"\nPROMPT: {t}")
            print(f"  area={r['area']} {r['unit']} | floors={r['floors']} | "
                  f"BHK={r['bhk']} | type={r['building_type']} | quality={r['quality']}")
            print(f"  city={r['city']} | state={r['state']}")
            for n in r['parsed_notes']:
                print(f"    → {n}")
        except ValueError as e:
            print(f"\nPROMPT: {t}\n  ERROR: {e}")