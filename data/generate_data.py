"""
Construction Material Dataset Generator
Generates synthetic training data based on standard Indian construction thumb rules
with realistic variance to simulate real-world project data.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

N = 5000  # number of training samples

def generate_dataset(n=N):
    # Area in sqft (100 to 10000 sqft)
    area_sqft = np.random.uniform(100, 10000, n)

    # Number of floors (1 to 4)
    floors = np.random.randint(1, 5, n)

    # Building type: 0=residential, 1=commercial, 2=industrial
    building_type = np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1])

    # Quality grade: 0=economy, 1=standard, 2=premium
    quality = np.random.choice([0, 1, 2], n, p=[0.3, 0.5, 0.2])

    total_area = area_sqft * floors

    # Quality multipliers
    q_mult = np.where(quality == 0, 0.85, np.where(quality == 1, 1.0, 1.2))

    # Building type multipliers
    bt_mult = np.where(building_type == 0, 1.0, np.where(building_type == 1, 1.15, 1.25))

    # ---- Thumb rules with noise ----
    # Cement: ~0.4 bags/sqft for standard RCC
    cement_bags = (total_area * 0.42 * q_mult * bt_mult
                   + np.random.normal(0, total_area * 0.01, n)).astype(int)

    # Sand: ~1.05 cft/sqft
    sand_cft = (total_area * 1.05 * q_mult
                + np.random.normal(0, total_area * 0.02, n)).astype(int)

    # Bricks: ~8 nos/sqft (not used in all structures, so commercial has less)
    brick_factor = np.where(building_type == 2, 5.0, 8.0)
    bricks = (total_area * brick_factor * q_mult
              + np.random.normal(0, total_area * 0.1, n)).astype(int)

    # Stone Aggregate: ~0.52 cft/sqft
    aggregate_cft = (total_area * 0.52 * q_mult * bt_mult
                     + np.random.normal(0, total_area * 0.015, n)).astype(int)

    # Steel: ~4.5 kg/sqft for residential, more for commercial
    steel_factor = np.where(building_type == 0, 4.5,
                   np.where(building_type == 1, 5.5, 6.5))
    steel_kg = (total_area * steel_factor * q_mult
                + np.random.normal(0, total_area * 0.05, n)).astype(int)

    # Clip negatives
    cement_bags = np.clip(cement_bags, 1, None)
    sand_cft = np.clip(sand_cft, 1, None)
    bricks = np.clip(bricks, 0, None)
    aggregate_cft = np.clip(aggregate_cft, 1, None)
    steel_kg = np.clip(steel_kg, 1, None)

    df = pd.DataFrame({
        "area_sqft": area_sqft.round(2),
        "floors": floors,
        "building_type": building_type,  # 0=residential,1=commercial,2=industrial
        "quality": quality,              # 0=economy,1=standard,2=premium
        "total_area_sqft": total_area.round(2),
        "cement_bags": cement_bags,
        "sand_cft": sand_cft,
        "bricks": bricks,
        "aggregate_cft": aggregate_cft,
        "steel_kg": steel_kg,
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("training_data.csv", index=False)
    print(f"Dataset generated: {len(df)} rows")
    print(df.describe())