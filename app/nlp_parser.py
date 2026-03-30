"""
NLP Parser — extracts structured construction details from free-form text prompts.

Examples of prompts it handles:
  "3 BHK 3 floor residential house"
  "I want to build a 2BHK house on 1200 sqft ground floor"
  "commercial building 5 floors 250 sqm premium quality"
  "small economy house 600 sqft"
"""

import re


# ── Lookup tables ──────────────────────────────────────────────────────────────

BHK_AREA_MAP = {
    1: 500,    # 1 BHK ~ 500 sqft per floor
    2: 850,    # 2 BHK ~ 850 sqft per floor
    3: 1200,   # 3 BHK ~ 1200 sqft per floor
    4: 1800,   # 4 BHK ~ 1800 sqft per floor
    5: 2400,   # 5 BHK ~ 2400 sqft per floor
}

BUILDING_TYPE_KEYWORDS = {
    0: ["residential", "house", "home", "villa", "bungalow", "flat",
        "apartment", "bhk", "duplex", "residence"],
    1: ["commercial", "office", "shop", "mall", "showroom", "hotel",
        "restaurant", "retail", "warehouse store"],
    2: ["industrial", "factory", "plant", "warehouse", "shed",
        "manufacturing", "workshop"],
}

QUALITY_KEYWORDS = {
    0: ["economy", "basic", "budget", "low cost", "simple", "ordinary",
        "affordable", "cheap"],
    1: ["standard", "normal", "regular", "moderate", "average", "typical"],
    2: ["premium", "luxury", "high end", "high-end", "deluxe", "superior",
        "top", "best", "executive"],
}

FLOOR_WORDS = {
    "ground": 1, "g": 1, "one": 1, "single": 1,
    "two": 2, "double": 2,
    "three": 3, "triple": 3,
    "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_prompt(text: str) -> dict:
    """
    Parse a free-form construction prompt into structured fields.

    Returns
    -------
    {
        area        : float,
        unit        : "sqft" | "sqm",
        floors      : int,
        building_type: int,   # 0/1/2
        quality     : int,    # 0/1/2
        bhk         : int | None,
        raw_prompt  : str,
        parsed_notes: list[str],   # human-readable parse trace
    }
    """
    raw   = text.strip()
    lower = raw.lower()
    notes = []

    # ── BHK ───────────────────────────────────────────────────────────────────
    bhk = None
    bhk_match = re.search(r'(\d)\s*bhk', lower)
    if bhk_match:
        bhk = int(bhk_match.group(1))
        notes.append(f"Detected {bhk} BHK")

    # ── Floors ────────────────────────────────────────────────────────────────
    floors = 1  # default

    # "G+2", "G + 3" patterns
    gplus = re.search(r'g\s*\+\s*(\d+)', lower)
    if gplus:
        floors = int(gplus.group(1)) + 1
        notes.append(f"Detected G+{gplus.group(1)} = {floors} floors")
    else:
        # "3 floor", "3 floors", "3-floor"
        floor_num = re.search(r'(\d+)\s*-?\s*floors?', lower)
        if floor_num:
            floors = min(int(floor_num.group(1)), 10)
            notes.append(f"Detected {floors} floors")
        else:
            # word form: "three floors"
            for word, val in FLOOR_WORDS.items():
                if re.search(rf'\b{word}\s+floors?\b', lower):
                    floors = val
                    notes.append(f"Detected floor word '{word}' = {floors}")
                    break

    # ── Area ──────────────────────────────────────────────────────────────────
    area = None
    unit = None

    # explicit area with unit
    area_match = re.search(
        r'(\d[\d,]*\.?\d*)\s*(sq\.?\s?ft|sqft|sq\.?\s?m|sqm|square\s?feet|square\s?meter)',
        lower
    )
    if area_match:
        area_raw = area_match.group(1).replace(",", "")
        area     = float(area_raw)
        unit_raw = area_match.group(2).replace(" ", "").replace(".", "")
        unit     = "sqm" if "sqm" in unit_raw or "squaremeter" in unit_raw else "sqft"
        notes.append(f"Detected area {area} {unit}")
    else:
        raise ValueError("Area unit (sqft or sqm) is mandatory. Please mention your area explicitly, e.g., '1200 sqft' or '250 sqm'.")

    # ── Building type ─────────────────────────────────────────────────────────
    # Check industrial first, then commercial, then residential (most specific → least)
    building_type = 0  # default residential
    for bt in [2, 1, 0]:
        keywords = BUILDING_TYPE_KEYWORDS[bt]
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', lower):
                building_type = bt
                notes.append(f"Building type → {['Residential','Commercial','Industrial'][bt]} (keyword: '{kw}')")
                break
        else:
            continue
        break

    # ── Quality ───────────────────────────────────────────────────────────────
    quality = 1  # default standard
    for q, keywords in QUALITY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                quality = q
                notes.append(f"Quality → {['Economy','Standard','Premium'][q]} (keyword: '{kw}')")
                break
        else:
            continue
        break

    return {
        "area":          area,
        "unit":          unit,
        "floors":        floors,
        "building_type": building_type,
        "quality":       quality,
        "bhk":           bhk,
        "raw_prompt":    raw,
        "parsed_notes":  notes,
    }


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "3 BHK 3 floor residential house",
        "I want to build a 2BHK house on 1200 sqft G+1",
        "commercial building 5 floors 250 sqm premium quality",
        "small economy house 600 sqft",
        "4 BHK luxury villa G+2",
        "industrial warehouse 5000 sqft 2 floors",
        "1 BHK apartment standard quality",
        "build a 3 bhk home",
    ]
    for s in samples:
        r = parse_prompt(s)
        print(f"\nPrompt : {s}")
        print(f"  area={r['area']} {r['unit']}, floors={r['floors']}, "
              f"type={r['building_type']}, quality={r['quality']}, bhk={r['bhk']}")
        for n in r['parsed_notes']:
            print(f"  → {n}")