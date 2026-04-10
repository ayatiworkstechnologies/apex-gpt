"""
Construction Material Dataset Generator -- v4 (Tier-Stratified, City-Aware)
=============================================================================
Generates SEPARATE datasets for Tier 1 / Tier 2 / Tier 3 Indian cities,
then MERGES them into a single training_data.csv.

Tier definitions:
  Tier 1 -- Major metros (Mumbai, Delhi, Bangalore, Hyderabad, Chennai,
             Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, ...)
             -> Larger buildings, more commercial/industrial mix,
                higher quality premium, live-verified prices.

  Tier 2 -- Mid-size cities (Nagpur, Indore, Coimbatore, Vijayawada, ...)
             -> Moderate-scale builds, balanced residential/commercial,
                regional-band pricing.

  Tier 3 -- Smaller cities (Erode, Salem, Thoothukudi, Tirunelveli, ...)
             -> Smaller residential builds dominate,
                economy/standard quality, lowest prices.

Samples per tier:
  Tier 1 : 12,000
  Tier 2 : 10,000
  Tier 3 :  5,000
  TOTAL   : 27,000

Thumb Rules (IS 456 / SP 16 / India 2026):
  Cement:    0.35-0.50 bags/sqft  ->  400-500 bags / 1000 sqft
  Sand:      1.2-1.8 cft/sqft     ->  1200-1800 cft / 1000 sqft
  Bricks:    5-8 nos/sqft         ->  5000-8000 / 1000 sqft
  Aggregate: 1.0-1.5 cft/sqft     ->  1000-1500 cft / 1000 sqft
  Steel:     3.5-5.5 kg/sqft      ->  3500-5500 kg / 1000 sqft
"""

import pandas as pd
import numpy as np
import os

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── Samples per tier ─────────────────────────────────────────────────────────
TIER_SAMPLES = {1: 12000, 2: 10000, 3: 5000}

# ── Tier-specific characteristics ────────────────────────────────────────────
# Area range (sqft)
TIER_AREA = {
    1: (500,  12000),   # metros: apartments to large commercial
    2: (300,   8000),   # mid-cities: houses to mid-rise
    3: (200,   4000),   # small cities: mostly individual houses
}

# Floor distribution (choices, probs)
TIER_FLOORS = {
    1: ([1, 2, 3, 4, 5, 6, 7, 8, 10], [0.10, 0.22, 0.22, 0.16, 0.10, 0.07, 0.05, 0.04, 0.04]),
    2: ([1, 2, 3, 4, 5, 6, 7, 8, 10], [0.22, 0.32, 0.22, 0.12, 0.06, 0.03, 0.01, 0.01, 0.01]),
    3: ([1, 2, 3, 4, 5],               [0.55, 0.30, 0.10, 0.03, 0.02]),
}

# Building type distribution  0=residential, 1=commercial, 2=industrial
TIER_BTYPE = {
    1: ([0, 1, 2], [0.50, 0.32, 0.18]),   # more commercial/industrial in metros
    2: ([0, 1, 2], [0.62, 0.25, 0.13]),
    3: ([0, 1, 2], [0.78, 0.16, 0.06]),   # mostly residential in small cities
}

# Quality distribution  0=economy, 1=standard, 2=premium
TIER_QUALITY = {
    1: ([0, 1, 2], [0.15, 0.45, 0.40]),   # metros: more premium
    2: ([0, 1, 2], [0.25, 0.52, 0.23]),
    3: ([0, 1, 2], [0.45, 0.45, 0.10]),   # small cities: mostly economy/standard
}

# Noise level sigma coefficient (higher = more variability)
TIER_NOISE = {1: 0.045, 2: 0.055, 3: 0.065}


# ── Load city rates ───────────────────────────────────────────────────────────
def load_city_rates(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = ["city", "state", "tier", "cost_mult", "labour_mult",
            "cement", "sand", "brick", "aggregate", "steel"]
    return df[cols].copy()


# ── Per-tier dataset generator ────────────────────────────────────────────────
def generate_tier_dataset(tier: int, n: int, city_df: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Generate n samples for a specific tier."""

    # Filter cities by tier
    tier_cities = city_df[city_df["tier"] == tier].reset_index(drop=True)
    if len(tier_cities) == 0:
        raise ValueError(f"No cities found for tier {tier}")

    n_cities = len(tier_cities)
    city_idx  = rng.integers(0, n_cities, n)

    city_names   = tier_cities["city"].values[city_idx]
    state_names  = tier_cities["state"].values[city_idx]
    cost_mult_c  = tier_cities["cost_mult"].values[city_idx].astype(float)

    up_cement    = tier_cities["cement"].values[city_idx].astype(float)
    up_sand      = tier_cities["sand"].values[city_idx].astype(float)
    up_brick     = tier_cities["brick"].values[city_idx].astype(float)
    up_aggregate = tier_cities["aggregate"].values[city_idx].astype(float)
    up_steel     = tier_cities["steel"].values[city_idx].astype(float)

    # ── Input features ───────────────────────────────────────────────────────
    lo, hi    = TIER_AREA[tier]
    area_sqft = rng.uniform(lo, hi, n)

    fl_choices, fl_probs = TIER_FLOORS[tier]
    floors = rng.choice(fl_choices, n, p=fl_probs)

    bt_choices, bt_probs = TIER_BTYPE[tier]
    building_type = rng.choice(bt_choices, n, p=bt_probs)

    q_choices, q_probs = TIER_QUALITY[tier]
    quality = rng.choice(q_choices, n, p=q_probs)

    total_area = area_sqft * floors

    # ── Multipliers ──────────────────────────────────────────────────────────
    q_mult = np.where(quality == 0, 0.82,
             np.where(quality == 1, 1.00, 1.22))

    bt_cement = np.where(building_type == 0, 1.00,
                np.where(building_type == 1, 1.12, 1.20))

    bt_steel = np.where(building_type == 0, 1.00,
               np.where(building_type == 1, 1.25, 1.45))

    bt_aggregate = np.where(building_type == 0, 1.00,
                   np.where(building_type == 1, 1.10, 1.18))

    # Foundation factor: +1.8% per floor over G+0, capped at +20%
    foundation_factor = 1.0 + np.clip((floors - 1) * 0.018, 0, 0.20)

    # Project complexity: partition walls, finishes (tier-3 less variable)
    complexity_range = {1: (0.90, 1.10), 2: (0.92, 1.08), 3: (0.94, 1.06)}
    lo_c, hi_c = complexity_range[tier]
    complexity = rng.uniform(lo_c, hi_c, n)

    # ── Noise (5-15% variability, higher for smaller cities) ────────────────
    sigma = TIER_NOISE[tier]
    noise_c  = rng.normal(0, total_area * sigma,       n)
    noise_s  = rng.normal(0, total_area * sigma * 1.1, n)
    noise_b  = rng.normal(0, total_area * sigma * 2.0, n)
    noise_ag = rng.normal(0, total_area * sigma * 1.0, n)
    noise_st = rng.normal(0, total_area * sigma * 1.2, n)

    # ── Material quantities (IS 456 / SP 16 thumb rules) ────────────────────
    # CEMENT: 0.42 bags/sqft base
    cement_bags = (
        total_area * 0.42 * q_mult * bt_cement * foundation_factor * complexity
        + noise_c
    )

    # SAND: 1.45 cft/sqft base
    sand_q = np.where(quality == 0, 0.88, np.where(quality == 1, 1.00, 1.15))
    sand_cft = (
        total_area * 1.45 * sand_q * complexity
        + noise_s
    )

    # BRICKS: 6.5 / 5.0 / 4.0 nos/sqft by type
    brick_base = np.where(building_type == 0, 6.5,
                 np.where(building_type == 1, 5.0, 4.0))
    brick_q = np.where(quality == 0, 0.90, np.where(quality == 1, 1.00, 1.10))
    bricks = (
        total_area * brick_base * brick_q * complexity
        + noise_b
    )

    # AGGREGATE: 1.20 cft/sqft base (corrected from v1's 0.52)
    aggregate_cft = (
        total_area * 1.20 * q_mult * bt_aggregate * foundation_factor * complexity
        + noise_ag
    )

    # STEEL: 4.0 / 5.0 / 5.8 kg/sqft by type (Fe500 TMT)
    steel_base = np.where(building_type == 0, 4.0,
                 np.where(building_type == 1, 5.0, 5.8))
    steel_q = np.where(quality == 0, 0.88, np.where(quality == 1, 1.00, 1.18))
    steel_kg = (
        total_area * steel_base * steel_q * bt_steel * foundation_factor * complexity
        + noise_st
    )

    # ── Clip & round ─────────────────────────────────────────────────────────
    cement_bags   = np.clip(cement_bags,   1, None).astype(int)
    sand_cft      = np.clip(sand_cft,      1, None).astype(int)
    bricks        = np.clip(bricks,        0, None).astype(int)
    aggregate_cft = np.clip(aggregate_cft, 1, None).astype(int)
    steel_kg      = np.clip(steel_kg,      1, None).astype(int)

    # ── Estimated costs (Rs) using city live unit prices ─────────────────────
    cost_cement    = cement_bags   * up_cement
    cost_sand      = sand_cft      * up_sand
    cost_bricks    = bricks        * up_brick
    cost_aggregate = aggregate_cft * up_aggregate
    cost_steel     = steel_kg      * up_steel
    total_cost     = cost_cement + cost_sand + cost_bricks + cost_aggregate + cost_steel

    return pd.DataFrame({
        # ── Inputs ──────────────────────────────────────────────────────────
        "city":            city_names,
        "state":           state_names,
        "tier":            tier,
        "area_sqft":       area_sqft.round(2),
        "floors":          floors,
        "building_type":   building_type,   # 0=residential 1=commercial 2=industrial
        "quality":         quality,          # 0=economy 1=standard 2=premium
        "total_area_sqft": total_area.round(2),
        # ── Quantities ──────────────────────────────────────────────────────
        "cement_bags":     cement_bags,
        "sand_cft":        sand_cft,
        "bricks":          bricks,
        "aggregate_cft":   aggregate_cft,
        "steel_kg":        steel_kg,
        # ── City unit prices (live April 2026) ───────────────────────────────
        "price_cement_per_bag": up_cement.astype(int),
        "price_sand_per_cft":   up_sand.astype(int),
        "price_brick_per_nos":  up_brick.astype(int),
        "price_agg_per_cft":    up_aggregate.astype(int),
        "price_steel_per_kg":   up_steel.astype(int),
        # ── Estimated costs (Rs) ─────────────────────────────────────────────
        "cost_cement":         cost_cement.round(0).astype(int),
        "cost_sand":           cost_sand.round(0).astype(int),
        "cost_bricks":         cost_bricks.round(0).astype(int),
        "cost_aggregate":      cost_aggregate.round(0).astype(int),
        "cost_steel":          cost_steel.round(0).astype(int),
        "total_material_cost": total_cost.round(0).astype(int),
    })


# ── Validation ────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame):
    print("\n--- Benchmark: ~1000 sqft | 1F | Residential | Standard ---")
    mask = (
        df["area_sqft"].between(950, 1050) &
        (df["floors"] == 1) &
        (df["building_type"] == 0) &
        (df["quality"] == 1)
    )
    s = df[mask]
    if len(s) == 0:
        print("  [WARN] No benchmark samples found.")
    else:
        print(f"  ({len(s)} samples across all tiers)")
        print(f"  Cement:    {s['cement_bags'].mean():.0f} bags     (expected: 400-500)")
        print(f"  Sand:      {s['sand_cft'].mean():.0f} cft      (expected: 1200-1800)")
        print(f"  Bricks:    {s['bricks'].mean():.0f} nos      (expected: 5000-8000)")
        print(f"  Aggregate: {s['aggregate_cft'].mean():.0f} cft      (expected: 1000-1500)")
        print(f"  Steel:     {s['steel_kg'].mean():.0f} kg       (expected: 3500-5500)")

    print("\n--- Samples, Cities, Cost per Tier ---")
    summary = df.groupby("tier").agg(
        samples=("tier", "count"),
        cities=("city", "nunique"),
        avg_area=("total_area_sqft", "mean"),
        avg_total_cost=("total_material_cost", "mean"),
    )
    summary["cost_per_sqft"] = (
        summary["avg_total_cost"] / summary["avg_area"]
    ).round(0).astype(int)
    summary["avg_total_cost"] = summary["avg_total_cost"].round(0).astype(int)
    print(summary.to_string())

    print("\n--- Global Per-1000-sqft Quantity Ratios ---")
    for mat, col in [("Cement (bags)", "cement_bags"),
                     ("Sand (cft)",    "sand_cft"),
                     ("Bricks (nos)",  "bricks"),
                     ("Agg (cft)",     "aggregate_cft"),
                     ("Steel (kg)",    "steel_kg")]:
        ratio = df[col] / df["total_area_sqft"] * 1000
        print(f"  {mat:<18s}  mean={ratio.mean():.0f}  "
              f"min={ratio.min():.0f}  max={ratio.max():.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir       = os.path.dirname(os.path.abspath(__file__))
    city_rates_csv = os.path.join(base_dir, "city_rates.csv")
    out_csv        = os.path.join(base_dir, "training_data.csv")

    city_df = load_city_rates(city_rates_csv)

    tier_counts = {t: len(city_df[city_df["tier"] == t]) for t in [1, 2, 3]}
    print(f"[INFO] City rates loaded: {len(city_df)} cities total")
    for t, cnt in tier_counts.items():
        print(f"       Tier {t}: {cnt} cities")

    # ── Generate per-tier datasets ───────────────────────────────────────────
    frames = []
    for tier, n_samples in TIER_SAMPLES.items():
        rng = np.random.default_rng(SEED + tier)     # reproducible per tier
        print(f"\n[GEN] Generating {n_samples:,} samples for Tier {tier} ...")
        df_tier = generate_tier_dataset(tier, n_samples, city_df, rng)
        frames.append(df_tier)
        print(f"      Done -- {df_tier['city'].nunique()} cities, "
              f"area {df_tier['area_sqft'].min():.0f}-{df_tier['area_sqft'].max():.0f} sqft")

    # ── Merge & shuffle ──────────────────────────────────────────────────────
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Save single CSV ──────────────────────────────────────────────────────
    df_all.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved -> {out_csv}")
    print(f"     Total rows : {len(df_all):,}")
    print(f"     Total cols : {len(df_all.columns)}")
    print(f"     Cities     : {df_all['city'].nunique()} "
          f"({sorted(df_all['tier'].unique())} tiers)")
    print(f"     File size  : {os.path.getsize(out_csv)/1024:.1f} KB")

    # ── Stats & validation ───────────────────────────────────────────────────
    print("\n--- Column Summary ---")
    print(df_all[["area_sqft", "floors", "building_type", "quality",
                  "cement_bags", "sand_cft", "bricks",
                  "aggregate_cft", "steel_kg",
                  "total_material_cost"]].describe().round(1).to_string())

    validate(df_all)

"""
v4 -- Tier-stratified generation:
  Tier 1: 12,000 samples | metros  | live April-2026 prices
  Tier 2: 10,000 samples | mid-cities | regional-band prices
  Tier 3:  5,000 samples | small cities | lowest price tier
  Total : 27,000 rows  | 24 columns | single training_data.csv
"""