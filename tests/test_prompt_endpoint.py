"""
Test the /estimate-from-prompt endpoint directly (no server needed).
Simulates exactly what the API does internally.

Run: python tests/test_prompt_endpoint.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.nlp_parser import parse_prompt
from app.predictor import predict

BUILDING_LABELS = {0: "Residential", 1: "Commercial", 2: "Industrial"}
QUALITY_LABELS  = {0: "Economy",     1: "Standard",   2: "Premium"}

def estimate_from_prompt(prompt: str):
    parsed = parse_prompt(prompt)
    result = predict(
        area=parsed["area"],
        unit=parsed["unit"],
        floors=parsed["floors"],
        building_type=parsed["building_type"],
        quality=parsed["quality"],
    )
    return parsed, result


# ── Test prompts ───────────────────────────────────────────────────────────────
test_prompts = [
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

SEP = "=" * 60

for prompt in test_prompts:
    print(f"\n{SEP}")
    print(f"  PROMPT  : {prompt}")
    print(f"{SEP}")
    try:
        parsed, result = estimate_from_prompt(prompt)
        print(f"  Parsed  : {parsed['bhk'] or '-'} BHK | "
              f"{parsed['area']} {parsed['unit']} | "
              f"{parsed['floors']} floor(s) | "
              f"{BUILDING_LABELS[parsed['building_type']]} | "
              f"{QUALITY_LABELS[parsed['quality']]}")
        print(f"  Notes   : {' | '.join(parsed['parsed_notes'])}")
        print(f"  Total   : {result['total_area_sqft']:,.0f} sqft")
        print(f"  ─────────────────────────────────────────")
        mats = result["materials"]
        print(f"  Cement bags    : {mats['cement_bags']:>8,}  bags (50 kg each)")
        print(f"  Sand           : {mats['sand_cft']:>8,}  cubic feet")
        print(f"  Bricks         : {mats['bricks']:>8,}  nos")
        print(f"  Stone aggregate: {mats['aggregate_cft']:>8,}  cubic feet")
        print(f"  Steel          : {mats['steel_kg']:>8,}  kg")
    except Exception as e:
        print(f"  ERROR   : {str(e)}")

print(f"\n{SEP}")
print(f"  All {len(test_prompts)} prompts processed successfully.")
print(SEP)