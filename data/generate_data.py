"""
Construction Material Dataset Generator — v2 (Real-World Calibrated)
====================================================================
Generates training data based on **validated Indian construction thumb rules**
sourced from civil engineering standards and industry practice (2024-2026).

Key improvements over v1:
  - Aggregate corrected to 1.0–1.5 cft/sqft (was 0.52 — too low)
  - Sand range widened to 1.2–1.8 cft/sqft (includes plastering/finishing)
  - Bricks adjusted to 6–8 avg (accounts for modern AAC/fly-ash mix)
  - Steel bands refined per IS code ranges
  - Higher noise (5–15%) to simulate real project variability
  - Floors range expanded to 1–10
  - Foundation factor added (multi-storey needs thicker foundation)
  - Room complexity factor (partitions, finishes)
  - 10,000 samples for better generalization

Thumb Rule Sources (India, 2026):
  Cement:    0.35–0.50 bags/sqft   →  400–500 bags per 1000 sqft
  Sand:      1.2–1.8 cft/sqft     →  1200–1800 cft per 1000 sqft
  Bricks:    5–8 nos/sqft          →  5000–8000 per 1000 sqft
  Aggregate: 1.0–1.5 cft/sqft     →  1000–1500 cft per 1000 sqft
  Steel:     3.5–5.5 kg/sqft      →  3500–5500 kg per 1000 sqft
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N = 10000  # doubled for better generalization


def generate_dataset(n=N):
    # ── Input features ─────────────────────────────────────────────────────

    # Area: 200 to 12,000 sqft (covers small houses to large commercial)
    area_sqft = np.random.uniform(200, 12000, n)

    # Floors: 1 to 10 (realistic Indian construction)
    floors = np.random.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 10],
        n,
        p=[0.20, 0.30, 0.20, 0.12, 0.07, 0.04, 0.03, 0.02, 0.02]
    )

    # Building type: 0=residential (60%), 1=commercial (25%), 2=industrial (15%)
    building_type = np.random.choice([0, 1, 2], n, p=[0.60, 0.25, 0.15])

    # Quality: 0=economy (25%), 1=standard (50%), 2=premium (25%)
    quality = np.random.choice([0, 1, 2], n, p=[0.25, 0.50, 0.25])

    total_area = area_sqft * floors

    # ── Multipliers ────────────────────────────────────────────────────────

    # Quality multiplier (economy uses less, premium uses more)
    q_mult = np.where(quality == 0, 0.82,
             np.where(quality == 1, 1.00, 1.22))

    # Building type multiplier (commercial/industrial need more cement & steel)
    bt_cement = np.where(building_type == 0, 1.00,
                np.where(building_type == 1, 1.12, 1.20))

    bt_steel = np.where(building_type == 0, 1.00,
               np.where(building_type == 1, 1.25, 1.45))

    bt_aggregate = np.where(building_type == 0, 1.00,
                   np.where(building_type == 1, 1.10, 1.18))

    # Foundation factor: multi-storey buildings need stronger foundation
    # G+0 = 1.0, G+1 = 1.02, G+2 = 1.05, G+3+ = progressively more
    foundation_factor = 1.0 + np.clip((floors - 1) * 0.018, 0, 0.20)

    # Room complexity: random per-project variation (wall partitions, finishes)
    complexity = np.random.uniform(0.92, 1.08, n)

    # ── Material calculations (industry thumb rules + noise) ───────────────

    # CEMENT: 0.38–0.48 bags/sqft (standard RCC residential)
    # Base: 0.42 bags/sqft
    cement_base = 0.42
    cement_bags = (total_area * cement_base * q_mult * bt_cement
                   * foundation_factor * complexity
                   + np.random.normal(0, total_area * 0.025, n))

    # SAND: 1.2–1.8 cft/sqft (mortar + plastering + concreting + finishing)
    # Base: 1.45 cft/sqft
    sand_base = 1.45
    sand_mult = np.where(quality == 0, 0.88,
                np.where(quality == 1, 1.00, 1.15))  # premium has more finishing
    sand_cft = (total_area * sand_base * sand_mult * complexity
                + np.random.normal(0, total_area * 0.035, n))

    # BRICKS: 5–8 nos/sqft depending on wall type
    # Residential: 6.5 avg (mix of 4.5" and 9" walls)
    # Commercial: 5.0 (more glass/curtain walls)
    # Industrial: 4.0 (more open spans, fewer partition walls)
    brick_base = np.where(building_type == 0, 6.5,
                 np.where(building_type == 1, 5.0, 4.0))
    brick_q = np.where(quality == 0, 0.90,
              np.where(quality == 1, 1.00, 1.10))  # premium has thicker walls
    bricks = (total_area * brick_base * brick_q * complexity
              + np.random.normal(0, total_area * 0.08, n))

    # AGGREGATE (20mm stone): 1.0–1.5 cft/sqft
    # Base: 1.20 cft/sqft (corrected from 0.52 — was severely underestimated)
    agg_base = 1.20
    aggregate_cft = (total_area * agg_base * q_mult * bt_aggregate
                     * foundation_factor * complexity
                     + np.random.normal(0, total_area * 0.03, n))

    # STEEL (Fe500): 3.5–5.5 kg/sqft
    # Residential: 4.0 kg/sqft
    # Commercial:  5.0 kg/sqft (more columns, longer spans)
    # Industrial:  5.8 kg/sqft (heavy duty, crane loads)
    steel_base = np.where(building_type == 0, 4.0,
                 np.where(building_type == 1, 5.0, 5.8))
    steel_q = np.where(quality == 0, 0.88,
              np.where(quality == 1, 1.00, 1.18))
    steel_kg = (total_area * steel_base * steel_q * foundation_factor
                * complexity
                + np.random.normal(0, total_area * 0.04, n))

    # ── Clip negatives & round ─────────────────────────────────────────────
    cement_bags   = np.clip(cement_bags, 1, None).astype(int)
    sand_cft      = np.clip(sand_cft, 1, None).astype(int)
    bricks        = np.clip(bricks, 0, None).astype(int)
    aggregate_cft = np.clip(aggregate_cft, 1, None).astype(int)
    steel_kg      = np.clip(steel_kg, 1, None).astype(int)

    df = pd.DataFrame({
        "area_sqft":       area_sqft.round(2),
        "floors":          floors,
        "building_type":   building_type,
        "quality":         quality,
        "total_area_sqft": total_area.round(2),
        "cement_bags":     cement_bags,
        "sand_cft":        sand_cft,
        "bricks":          bricks,
        "aggregate_cft":   aggregate_cft,
        "steel_kg":        steel_kg,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset generated: {len(df)} rows → {out_path}")
    print(f"\n── Summary Statistics ──")
    print(df.describe().round(1))

    # Validate: 1000 sqft, 1 floor, residential, standard
    mask = (
        (df["area_sqft"].between(950, 1050)) &
        (df["floors"] == 1) &
        (df["building_type"] == 0) &
        (df["quality"] == 1)
    )
    sample = df[mask]
    if len(sample) > 0:
        print(f"\n── Validation: ~1000 sqft, 1F, Residential, Standard ({len(sample)} samples) ──")
        print(f"  Cement:    {sample['cement_bags'].mean():.0f} bags   (expected: 400–500)")
        print(f"  Sand:      {sample['sand_cft'].mean():.0f} cft    (expected: 1200–1800)")
        print(f"  Bricks:    {sample['bricks'].mean():.0f} nos    (expected: 5000–8000)")
        print(f"  Aggregate: {sample['aggregate_cft'].mean():.0f} cft    (expected: 1000–1500)")
        print(f"  Steel:     {sample['steel_kg'].mean():.0f} kg     (expected: 3500–4500)")
"""
Description:    Upgraded data generator with real-world calibrated formulas
"""