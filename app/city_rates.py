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

# CITY_DB is populated from data/city_rates.csv at import time (see
# _load_city_rates_from_csv below), which is the single source of truth for
# rates. This fallback is only used if the CSV is missing, so it carries one
# verified entry (Chennai, kept in sync with the CSV) rather than a full city
# list that would otherwise drift out of sync with the CSV over time.
DEFAULT_CITY_DB = {
    "chennai": {
        "state": "Tamil Nadu", "tier": 1,
        "cost_mult": 1.05, "labour_mult": 1.05,
        "cement": 410, "sand": 73, "brick": 8, "aggregate": 45, "steel": 61,
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


def reload_city_db() -> dict:
    """Reload city rates from CSV so runtime uses the latest fetched prices."""
    global CITY_DB
    CITY_DB = _load_city_rates_from_csv(DEFAULT_CITY_DB)
    return CITY_DB


def resolve_city(city_raw: Optional[str]) -> dict:
    """
    Resolve raw city/state string to a city record.
    Returns the city data dict with all rate info.
    Falls back to Chennai (baseline) if not found.
    """
    key = (city_raw or "").lower().strip().replace("-", " ")
    if not key:
        return {"city": "Chennai (default)", **CITY_DB["chennai"]}

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


def _format_range(low: float, high: float) -> str:
    return f"{int(round(low))}-{int(round(high))}"


def _scaled_ton_range_to_unit_range(rate: int, low_ton: float, high_ton: float) -> str:
    midpoint = (low_ton + high_ton) / 2
    if midpoint <= 0:
        return ""
    return _format_range(rate * (low_ton / midpoint), rate * (high_ton / midpoint))


def _fallback_range(rate: int) -> str:
    return _format_range(rate * 0.9, rate * 1.1)


def _extract_ranges_from_notes(notes: str, rates: Optional[dict] = None) -> dict:
    text = notes or ""
    import re

    material_patterns = {
        "cement_per_bag": (r"cement", "cement_per_bag"),
        "sand_per_cft": (r"sand|m-sand|river sand", "sand_per_cft"),
        "brick_per_nos": (r"brick", "brick_per_nos"),
        "aggregate_per_cft": (r"aggregate|jelly", "aggregate_per_cft"),
        "steel_per_kg": (r"steel", "steel_per_kg"),
    }
    ranges = {}

    for key, (material_pattern, rate_key) in material_patterns.items():
        material_match = re.search(
            rf"(?:{material_pattern}).{{0,120}}?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?:\s*/\s*(ton|kg|cft|nos|bag))?",
            text,
            re.IGNORECASE,
        )
        if not material_match:
            continue

        low = float(material_match.group(1))
        high = float(material_match.group(2))
        unit = (material_match.group(3) or "").lower()
        rate = int((rates or {}).get(rate_key, 0) or 0)

        if unit == "ton" and rate and key in {"sand_per_cft", "aggregate_per_cft"}:
            ranges[key] = _scaled_ton_range_to_unit_range(rate, low, high)
        elif unit == "ton" and key == "steel_per_kg":
            ranges[key] = _format_range(low / 1000, high / 1000)
        elif key == "steel_per_kg" and low > 1000 and high > 1000:
            ranges[key] = _format_range(low / 1000, high / 1000)
        elif key == "aggregate_per_cft" and low == 20:
            continue
        else:
            ranges[key] = _format_range(low, high)

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

    rates_used = {
        "cement_per_bag":    city["cement"],
        "sand_per_cft":      city["sand"],
        "brick_per_nos":     city["brick"],
        "aggregate_per_cft": city["aggregate"],
        "steel_per_kg":      city["steel"],
    }
    rate_ranges = _extract_ranges_from_notes(city.get("notes", ""), rates_used)
    for key, value in rates_used.items():
        rate_ranges.setdefault(key, _fallback_range(value))

    return {
        "city":           city["city"],
        "state":          city["state"],
        "tier":           city.get("tier", 2),
        "rates_used":     rates_used,
        "rate_ranges":    rate_ranges,
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
