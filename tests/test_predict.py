"""
Direct inference tests — no server required.
Run: python tests/test_predict.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.predictor import predict

cases = [
    # (description, payload)
    ("Minimal — 1000 sqft residential standard 1F",
     dict(area=1000, unit="sqft", floors=1, building_type=0, quality=1)),

    ("Commercial — 250 sqm, 3 floors, premium",
     dict(area=250,  unit="sqm",  floors=3, building_type=1, quality=2)),

    ("Economy residential — 600 sqft ground floor",
     dict(area=600,  unit="sqft", floors=1, building_type=0, quality=0)),

    ("Industrial — 5000 sqft, 4 floors, standard",
     dict(area=5000, unit="sqft", floors=4, building_type=2, quality=1)),
]

for i, (desc, payload) in enumerate(cases, 1):
    result = predict(**payload)
    print(f"\n── Test {i}: {desc} ──")
    print(f"  Total area   : {result['total_area_sqft']:>10,.0f} sqft")
    print(f"  Building     : {result['building_type']}  |  Quality: {result['quality']}")
    print(f"  ─────────────────────────────────────")
    for mat, qty in result["materials"].items():
        print(f"  {mat:<22}  {qty:>10,}")
