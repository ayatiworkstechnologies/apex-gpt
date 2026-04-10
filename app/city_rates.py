"""
City & State based construction cost database.
Contains material rate multipliers and per-unit cost (₹) for Indian cities.

Base rates (Standard quality, Residential, Chennai):
  Cement  : ₹420 / bag (50kg)
  Sand    : ₹55  / cft
  Bricks  : ₹9   / nos
  Aggregate: ₹60  / cft
  Steel   : ₹72  / kg
"""

import csv
import os
from typing import Optional


DEFAULT_RATE_META = {
    "verified": False,
    "last_updated": "",
    "source_url": "",
    "source_label": "baseline",
    "notes": "Default baseline rate. Replace with supplier quote or verified market data.",
}

# ── Base material rates (₹) — Chennai as baseline ─────────────────────────────
BASE_RATES = {
    "cement_per_bag":    420,   # ₹ per 50kg bag
    "sand_per_cft":       55,   # ₹ per cubic foot
    "brick_per_nos":       9,   # ₹ per brick
    "aggregate_per_cft":  60,   # ₹ per cubic foot
    "steel_per_kg":       72,   # ₹ per kg (Fe500)
}

# ── City database ──────────────────────────────────────────────────────────────
# cost_mult : overall construction cost multiplier vs Chennai baseline
# labour_mult: labour cost multiplier
# cement_rate, sand_rate, brick_rate, aggregate_rate, steel_rate: local ₹ rates

CITY_RATES_CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/city_rates.csv")

DEFAULT_CITY_DB = {
    # ── Tamil Nadu ──────────────────────────────────────────────────────────
    "chennai": {
        "state": "Tamil Nadu", "tier": 1,
        "cost_mult": 1.00, "labour_mult": 1.00,
        "cement": 420, "sand": 55,  "brick": 9,  "aggregate": 60, "steel": 72,
    },
    "coimbatore": {
        "state": "Tamil Nadu", "tier": 2,
        "cost_mult": 0.90, "labour_mult": 0.88,
        "cement": 400, "sand": 48,  "brick": 8,  "aggregate": 55, "steel": 70,
    },
    "madurai": {
        "state": "Tamil Nadu", "tier": 2,
        "cost_mult": 0.88, "labour_mult": 0.85,
        "cement": 395, "sand": 45,  "brick": 7,  "aggregate": 52, "steel": 69,
    },
    "trichy": {
        "state": "Tamil Nadu", "tier": 2,
        "cost_mult": 0.87, "labour_mult": 0.84,
        "cement": 390, "sand": 44,  "brick": 7,  "aggregate": 51, "steel": 68,
    },
    "salem": {
        "state": "Tamil Nadu", "tier": 2,
        "cost_mult": 0.86, "labour_mult": 0.83,
        "cement": 388, "sand": 43,  "brick": 7,  "aggregate": 50, "steel": 68,
    },
    "tirunelveli": {
        "state": "Tamil Nadu", "tier": 3,
        "cost_mult": 0.84, "labour_mult": 0.80,
        "cement": 380, "sand": 42,  "brick": 6,  "aggregate": 48, "steel": 67,
    },
    "erode": {
        "state": "Tamil Nadu", "tier": 3,
        "cost_mult": 0.85, "labour_mult": 0.82,
        "cement": 382, "sand": 42,  "brick": 7,  "aggregate": 49, "steel": 67,
    },
    "vellore": {
        "state": "Tamil Nadu", "tier": 3,
        "cost_mult": 0.85, "labour_mult": 0.82,
        "cement": 385, "sand": 43,  "brick": 7,  "aggregate": 50, "steel": 68,
    },

    # ── Maharashtra ─────────────────────────────────────────────────────────
    "mumbai": {
        "state": "Maharashtra", "tier": 1,
        "cost_mult": 1.45, "labour_mult": 1.50,
        "cement": 480, "sand": 90,  "brick": 14, "aggregate": 95, "steel": 78,
    },
    "pune": {
        "state": "Maharashtra", "tier": 1,
        "cost_mult": 1.25, "labour_mult": 1.28,
        "cement": 455, "sand": 75,  "brick": 12, "aggregate": 80, "steel": 76,
    },
    "nagpur": {
        "state": "Maharashtra", "tier": 2,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 73,
    },
    "nashik": {
        "state": "Maharashtra", "tier": 2,
        "cost_mult": 1.08, "labour_mult": 1.05,
        "cement": 435, "sand": 62,  "brick": 10, "aggregate": 67, "steel": 73,
    },
    "aurangabad": {
        "state": "Maharashtra", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.98,
        "cement": 425, "sand": 58,  "brick": 9,  "aggregate": 62, "steel": 72,
    },

    # ── Karnataka ───────────────────────────────────────────────────────────
    "bangalore": {
        "state": "Karnataka", "tier": 1,
        "cost_mult": 1.30, "labour_mult": 1.35,
        "cement": 460, "sand": 80,  "brick": 13, "aggregate": 85, "steel": 76,
    },
    "mysore": {
        "state": "Karnataka", "tier": 2,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 58,  "brick": 10, "aggregate": 63, "steel": 73,
    },
    "hubli": {
        "state": "Karnataka", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.97,
        "cement": 422, "sand": 56,  "brick": 9,  "aggregate": 61, "steel": 72,
    },
    "mangalore": {
        "state": "Karnataka", "tier": 2,
        "cost_mult": 1.02, "labour_mult": 0.99,
        "cement": 425, "sand": 57,  "brick": 9,  "aggregate": 62, "steel": 72,
    },

    # ── Andhra Pradesh & Telangana ───────────────────────────────────────────
    "hyderabad": {
        "state": "Telangana", "tier": 1,
        "cost_mult": 1.20, "labour_mult": 1.22,
        "cement": 448, "sand": 68,  "brick": 11, "aggregate": 74, "steel": 75,
    },
    "visakhapatnam": {
        "state": "Andhra Pradesh", "tier": 1,
        "cost_mult": 1.10, "labour_mult": 1.08,
        "cement": 438, "sand": 63,  "brick": 10, "aggregate": 68, "steel": 74,
    },
    "vijayawada": {
        "state": "Andhra Pradesh", "tier": 2,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 73,
    },
    "guntur": {
        "state": "Andhra Pradesh", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.98,
        "cement": 422, "sand": 57,  "brick": 9,  "aggregate": 62, "steel": 72,
    },
    "warangal": {
        "state": "Telangana", "tier": 2,
        "cost_mult": 0.98, "labour_mult": 0.95,
        "cement": 418, "sand": 55,  "brick": 9,  "aggregate": 60, "steel": 71,
    },

    # ── Delhi NCR ───────────────────────────────────────────────────────────
    "delhi": {
        "state": "Delhi", "tier": 1,
        "cost_mult": 1.40, "labour_mult": 1.42,
        "cement": 475, "sand": 88,  "brick": 13, "aggregate": 92, "steel": 78,
    },
    "gurgaon": {
        "state": "Haryana", "tier": 1,
        "cost_mult": 1.42, "labour_mult": 1.45,
        "cement": 478, "sand": 90,  "brick": 14, "aggregate": 94, "steel": 79,
    },
    "noida": {
        "state": "Uttar Pradesh", "tier": 1,
        "cost_mult": 1.38, "labour_mult": 1.40,
        "cement": 472, "sand": 86,  "brick": 13, "aggregate": 90, "steel": 77,
    },
    "ghaziabad": {
        "state": "Uttar Pradesh", "tier": 2,
        "cost_mult": 1.25, "labour_mult": 1.28,
        "cement": 455, "sand": 75,  "brick": 12, "aggregate": 80, "steel": 76,
    },
    "faridabad": {
        "state": "Haryana", "tier": 2,
        "cost_mult": 1.22, "labour_mult": 1.25,
        "cement": 450, "sand": 72,  "brick": 11, "aggregate": 77, "steel": 75,
    },

    # ── Gujarat ──────────────────────────────────────────────────────────────
    "ahmedabad": {
        "state": "Gujarat", "tier": 1,
        "cost_mult": 1.15, "labour_mult": 1.12,
        "cement": 443, "sand": 66,  "brick": 11, "aggregate": 71, "steel": 74,
    },
    "surat": {
        "state": "Gujarat", "tier": 1,
        "cost_mult": 1.12, "labour_mult": 1.10,
        "cement": 440, "sand": 64,  "brick": 10, "aggregate": 69, "steel": 74,
    },
    "vadodara": {
        "state": "Gujarat", "tier": 2,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 73,
    },
    "rajkot": {
        "state": "Gujarat", "tier": 2,
        "cost_mult": 1.02, "labour_mult": 0.99,
        "cement": 425, "sand": 58,  "brick": 9,  "aggregate": 63, "steel": 72,
    },

    # ── Rajasthan ────────────────────────────────────────────────────────────
    "jaipur": {
        "state": "Rajasthan", "tier": 1,
        "cost_mult": 1.08, "labour_mult": 1.05,
        "cement": 435, "sand": 62,  "brick": 10, "aggregate": 67, "steel": 73,
    },
    "jodhpur": {
        "state": "Rajasthan", "tier": 2,
        "cost_mult": 0.98, "labour_mult": 0.95,
        "cement": 418, "sand": 55,  "brick": 9,  "aggregate": 60, "steel": 71,
    },
    "udaipur": {
        "state": "Rajasthan", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.97,
        "cement": 422, "sand": 57,  "brick": 9,  "aggregate": 62, "steel": 71,
    },

    # ── West Bengal ──────────────────────────────────────────────────────────
    "kolkata": {
        "state": "West Bengal", "tier": 1,
        "cost_mult": 1.18, "labour_mult": 1.15,
        "cement": 445, "sand": 67,  "brick": 11, "aggregate": 72, "steel": 75,
    },
    "howrah": {
        "state": "West Bengal", "tier": 1,
        "cost_mult": 1.15, "labour_mult": 1.12,
        "cement": 442, "sand": 65,  "brick": 11, "aggregate": 70, "steel": 74,
    },
    "durgapur": {
        "state": "West Bengal", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.98,
        "cement": 422, "sand": 57,  "brick": 9,  "aggregate": 62, "steel": 72,
    },

    # ── Kerala ───────────────────────────────────────────────────────────────
    "kochi": {
        "state": "Kerala", "tier": 1,
        "cost_mult": 1.22, "labour_mult": 1.28,
        "cement": 450, "sand": 72,  "brick": 11, "aggregate": 77, "steel": 75,
    },
    "thiruvananthapuram": {
        "state": "Kerala", "tier": 1,
        "cost_mult": 1.18, "labour_mult": 1.22,
        "cement": 445, "sand": 68,  "brick": 11, "aggregate": 73, "steel": 74,
    },
    "kozhikode": {
        "state": "Kerala", "tier": 2,
        "cost_mult": 1.10, "labour_mult": 1.14,
        "cement": 437, "sand": 63,  "brick": 10, "aggregate": 68, "steel": 73,
    },

    # ── Punjab & Haryana ─────────────────────────────────────────────────────
    "chandigarh": {
        "state": "Punjab", "tier": 1,
        "cost_mult": 1.20, "labour_mult": 1.18,
        "cement": 447, "sand": 68,  "brick": 11, "aggregate": 73, "steel": 75,
    },
    "ludhiana": {
        "state": "Punjab", "tier": 1,
        "cost_mult": 1.12, "labour_mult": 1.10,
        "cement": 438, "sand": 64,  "brick": 10, "aggregate": 69, "steel": 73,
    },
    "amritsar": {
        "state": "Punjab", "tier": 2,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 72,
    },

    # ── Uttar Pradesh ────────────────────────────────────────────────────────
    "lucknow": {
        "state": "Uttar Pradesh", "tier": 1,
        "cost_mult": 1.10, "labour_mult": 1.08,
        "cement": 437, "sand": 63,  "brick": 10, "aggregate": 68, "steel": 73,
    },
    "kanpur": {
        "state": "Uttar Pradesh", "tier": 1,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 72,
    },
    "agra": {
        "state": "Uttar Pradesh", "tier": 2,
        "cost_mult": 1.00, "labour_mult": 0.98,
        "cement": 422, "sand": 57,  "brick": 9,  "aggregate": 62, "steel": 71,
    },
    "varanasi": {
        "state": "Uttar Pradesh", "tier": 2,
        "cost_mult": 0.98, "labour_mult": 0.95,
        "cement": 418, "sand": 55,  "brick": 9,  "aggregate": 60, "steel": 71,
    },

    # ── Madhya Pradesh ───────────────────────────────────────────────────────
    "bhopal": {
        "state": "Madhya Pradesh", "tier": 1,
        "cost_mult": 1.02, "labour_mult": 0.99,
        "cement": 425, "sand": 58,  "brick": 9,  "aggregate": 63, "steel": 72,
    },
    "indore": {
        "state": "Madhya Pradesh", "tier": 1,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 72,
    },
    "jabalpur": {
        "state": "Madhya Pradesh", "tier": 2,
        "cost_mult": 0.95, "labour_mult": 0.92,
        "cement": 412, "sand": 52,  "brick": 8,  "aggregate": 57, "steel": 70,
    },

    # ── Odisha ───────────────────────────────────────────────────────────────
    "bhubaneswar": {
        "state": "Odisha", "tier": 1,
        "cost_mult": 1.05, "labour_mult": 1.02,
        "cement": 430, "sand": 60,  "brick": 10, "aggregate": 65, "steel": 72,
    },
    "cuttack": {
        "state": "Odisha", "tier": 2,
        "cost_mult": 0.98, "labour_mult": 0.95,
        "cement": 418, "sand": 55,  "brick": 9,  "aggregate": 60, "steel": 71,
    },

    # ── Assam & Northeast ────────────────────────────────────────────────────
    "guwahati": {
        "state": "Assam", "tier": 1,
        "cost_mult": 1.12, "labour_mult": 1.10,
        "cement": 448, "sand": 65,  "brick": 10, "aggregate": 70, "steel": 74,
    },
}

# ── Aliases & alternate spellings ──────────────────────────────────────────────
CITY_ALIASES = {
    "bengaluru": "bangalore", "blr": "bangalore", "bengalore": "bangalore",
    "bombay": "mumbai", "bom": "mumbai",
    "calcutta": "kolkata", "cal": "kolkata",
    "madras": "chennai", "chn": "chennai",
    "hyd": "hyderabad", "vizag": "visakhapatnam",
    "trivandrum": "thiruvananthapuram",
    "trichy": "trichy", "tiruchirappalli": "trichy",
    "navi mumbai": "mumbai", "thane": "mumbai",
    "gurugram": "gurgaon",
    "new delhi": "delhi",
    "cochin": "kochi",
}

# ── State defaults (used when only state is mentioned) ─────────────────────────
STATE_DEFAULT_CITY = {
    "Tamil Nadu": "chennai",
    "Maharashtra": "pune",
    "Karnataka": "bangalore",
    "Telangana": "hyderabad",
    "Andhra Pradesh": "vijayawada",
    "Delhi": "delhi",
    "Gujarat": "ahmedabad",
    "Rajasthan": "jaipur",
    "West Bengal": "kolkata",
    "Kerala": "kochi",
    "Punjab": "chandigarh",
    "Haryana": "gurgaon",
    "Uttar Pradesh": "lucknow",
    "Madhya Pradesh": "indore",
    "Odisha": "bhubaneswar",
    "Assam": "guwahati",
}

STATE_ALIASES = {
    "tn": "Tamil Nadu", "tamilnadu": "Tamil Nadu",
    "mh": "Maharashtra", "maha": "Maharashtra",
    "ka": "Karnataka", "karnataka": "Karnataka",
    "ts": "Telangana", "tg": "Telangana",
    "ap": "Andhra Pradesh", "andhrapradesh": "Andhra Pradesh",
    "dl": "Delhi", "ncr": "Delhi",
    "gj": "Gujarat",
    "rj": "Rajasthan",
    "wb": "West Bengal", "westbengal": "West Bengal",
    "kl": "Kerala",
    "pb": "Punjab",
    "hr": "Haryana",
    "up": "Uttar Pradesh", "uttarpradesh": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "madhyapradesh": "Madhya Pradesh",
    "od": "Odisha", "or": "Odisha",
    "as": "Assam",
}


def _with_rate_meta(record: dict) -> dict:
    merged = DEFAULT_RATE_META.copy()
    merged.update(record)
    merged["verified"] = str(merged.get("verified", False)).strip().lower() in {
        "1", "true", "yes", "y"
    } if not isinstance(merged.get("verified"), bool) else merged["verified"]
    return merged


def _load_city_rates_from_csv(default_db: dict) -> dict:
    if not os.path.exists(CITY_RATES_CSV_PATH):
        return {key: _with_rate_meta(value.copy()) for key, value in default_db.items()}

    merged = {key: _with_rate_meta(value.copy()) for key, value in default_db.items()}
    with open(CITY_RATES_CSV_PATH, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            city = (row.get("city") or "").strip().lower()
            state = (row.get("state") or "").strip()
            if not city or not state:
                continue

            merged[city] = _with_rate_meta({
                "state": state,
                "tier": int(float(row.get("tier") or 2)),
                "cost_mult": float(row.get("cost_mult") or 1.0),
                "labour_mult": float(row.get("labour_mult") or 1.0),
                "cement": int(float(row.get("cement") or 420)),
                "sand": int(float(row.get("sand") or 55)),
                "brick": int(float(row.get("brick") or 9)),
                "aggregate": int(float(row.get("aggregate") or 60)),
                "steel": int(float(row.get("steel") or 72)),
                "verified": row.get("verified", ""),
                "last_updated": (row.get("last_updated") or "").strip(),
                "source_url": (row.get("source_url") or "").strip(),
                "source_label": (row.get("source_label") or "").strip() or "csv",
                "notes": (row.get("notes") or "").strip(),
            })
    return merged


CITY_DB = _load_city_rates_from_csv(DEFAULT_CITY_DB)


def resolve_city(city_raw: Optional[str]) -> dict:
    """
    Resolve raw city/state string to a city record.
    Returns the city data dict with all rate info.
    Falls back to Chennai (baseline) if not found.
    """
    if not city_raw:
        return {"city": "Chennai (default)", **CITY_DB["chennai"]}

    key = city_raw.lower().strip().replace("-", " ")

    # direct match
    if key in CITY_DB:
        return {"city": key.title(), **CITY_DB[key]}

    # alias match
    if key in CITY_ALIASES:
        resolved = CITY_ALIASES[key]
        return {"city": resolved.title(), **CITY_DB[resolved]}

    # state → default city
    state_norm = STATE_ALIASES.get(key.replace(" ", "").lower())
    if not state_norm:
        # try partial state name
        for sname, scity in STATE_DEFAULT_CITY.items():
            if key in sname.lower():
                state_norm = sname
                break

    if state_norm and state_norm in STATE_DEFAULT_CITY:
        default_city = STATE_DEFAULT_CITY[state_norm]
        return {"city": f"{default_city.title()} ({state_norm})", **CITY_DB[default_city]}

    # fuzzy partial match on city names
    for cname in CITY_DB:
        if key in cname or cname in key:
            return {"city": cname.title(), **CITY_DB[cname]}

    # fallback
    return {"city": f"{city_raw.title()} (not found — using Chennai rates)", **CITY_DB["chennai"]}


def get_all_cities() -> list:
    return sorted(CITY_DB.keys())


def get_city_rate_stats() -> dict:
    total = len(CITY_DB)
    verified = sum(1 for row in CITY_DB.values() if row.get("verified"))
    return {
        "total": total,
        "verified": verified,
        "unverified": total - verified,
    }


def _extract_ranges_from_notes(notes: str) -> dict:
    text = notes or ""
    patterns = {
        "cement_per_bag": r"cement.*?\(?(\d+(?:-\d+)?)\)?",
        "sand_per_cft": r"sand.*?\(?(\d+(?:-\d+)?)\)?",
        "brick_per_nos": r"brick.*?\(?(\d+(?:-\d+)?)\)?",
        "aggregate_per_cft": r"(?:aggregate|jelly).*?\(?(\d+(?:-\d+)?)\)?",
        "steel_per_kg": r"steel.*?\(?(\d+(?:-\d+)?)\)?",
    }
    import re
    ranges = {}
    for key, pattern in patterns.items():
        # Specifically look for (low-high) format or similar near the material mention
        match = re.search(r"(" + key.split("_")[0] + r".*?)\((\d+-\d+)\)", text, re.IGNORECASE)
        if match:
            # Found (100-200) style
            ranges[key] = match.group(2)
        else:
            # Fallback legacy parsing
            match2 = re.search(pattern, text, re.IGNORECASE)
            if match2:
                # Discard silly 20mm matches for aggregate
                if key == "aggregate_per_cft" and match2.group(1) == "20":
                    match3 = re.search(r"aggregate 20mm.*?\((\d+-\d+)\)", text, re.IGNORECASE)
                    if match3:
                        ranges[key] = match3.group(1)
                else:
                    ranges[key] = match2.group(1)
    return ranges


def get_cost_estimate(materials: dict, city_key: str = "chennai",
                      total_sqft: float = 0, quality: int = 1) -> dict:
    """
    Given material quantities, city, area, and quality — return full ₹ cost estimate.

    Real Indian construction cost breakdown (2026):
      Structural materials : ~28-32% of total
      Labour (all trades)  : ~28-32% of total
      Finishing & MEP      : ~25-30% of total  (flooring, paint, plumbing,
                             electrical, doors, windows, waterproofing)
      Overhead & misc      : ~8-12% of total   (scaffolding, permits,
                             transport, water, supervision)

    Target cost/sqft (Chennai baseline):
      Economy  : ₹1,700 – ₹2,000 / sqft
      Standard : ₹2,000 – ₹2,500 / sqft
      Premium  : ₹2,500 – ₹3,500 / sqft
    """
    city = resolve_city(city_key)

    # ── 1. Structural material cost ────────────────────────────────────────
    breakdown = {
        "cement":    materials["cement_bags"]   * city["cement"],
        "sand":      materials["sand_cft"]       * city["sand"],
        "bricks":    materials["bricks"]         * city["brick"],
        "aggregate": materials["aggregate_cft"]  * city["aggregate"],
        "steel":     materials["steel_kg"]       * city["steel"],
    }
    material_total = sum(breakdown.values())

    # ── 2. Labour cost (all trades: masonry, carpentry, bar-bending,
    #        plumbing labour, electrical labour, painting labour) ───────────
    # Labour ≈ 90-110% of structural material cost, varies by city
    labour_mult_base = {0: 0.88, 1: 1.00, 2: 1.15}  # economy/standard/premium
    labour_cost = int(material_total * labour_mult_base.get(quality, 1.0)
                      * city["labour_mult"])

    # ── 3. Finishing & MEP cost ────────────────────────────────────────────
    # Includes: flooring/tiles, interior/exterior paint, doors & windows,
    #           plumbing fixtures, electrical fixtures & wiring, waterproofing
    # Rate per sqft varies by quality:
    #   Economy  : ₹350–450 / sqft
    #   Standard : ₹500–650 / sqft
    #   Premium  : ₹800–1100 / sqft
    finishing_rate = {0: 400, 1: 575, 2: 950}
    finishing_cost = int(total_sqft * finishing_rate.get(quality, 575)
                        * city["cost_mult"]) if total_sqft else 0

    # ── 4. Overhead & miscellaneous ────────────────────────────────────────
    # Scaffolding, transport, water supply, permits, supervision, contingency
    # ~10% of (material + labour + finishing)
    subtotal = material_total + labour_cost + finishing_cost
    overhead_cost = int(subtotal * 0.10)

    total = material_total + labour_cost + finishing_cost + overhead_cost

    return {
        "city":           city["city"],
        "state":          city["state"],
        "tier":           city.get("tier", 2),
        "rates_used":     {
            "cement_per_bag":    city["cement"],
            "sand_per_cft":      city["sand"],
            "brick_per_nos":     city["brick"],
            "aggregate_per_cft": city["aggregate"],
            "steel_per_kg":      city["steel"],
        },
        "rate_ranges":    _extract_ranges_from_notes(city.get("notes", "")),
        "rate_meta":      {
            "verified": city.get("verified", False),
            "last_updated": city.get("last_updated", ""),
            "source_label": city.get("source_label", ""),
            "source_url": city.get("source_url", ""),
            "notes": city.get("notes", ""),
        },
        "cost_breakdown":  {k: int(v) for k, v in breakdown.items()},
        "material_total":  int(material_total),
        "labour_cost":     labour_cost,
        "finishing_cost":  finishing_cost,
        "overhead_cost":   overhead_cost,
        "total_cost_inr":  int(total),
        "cost_per_sqft":   0,   # filled by caller after total_sqft is known
    }
