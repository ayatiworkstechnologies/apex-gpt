"""Quick runner — writes training_data.csv to a fresh file name to avoid lock issues."""
import pandas as pd
import numpy as np
import os

SEED = 42
TIER_SAMPLES = {1: 12000, 2: 10000, 3: 5000}
TIER_AREA    = {1:(500,12000), 2:(300,8000), 3:(200,4000)}
TIER_FLOORS  = {
    1: ([1,2,3,4,5,6,7,8,10],[.10,.22,.22,.16,.10,.07,.05,.04,.04]),
    2: ([1,2,3,4,5,6,7,8,10],[.22,.32,.22,.12,.06,.03,.01,.01,.01]),
    3: ([1,2,3,4,5],          [.55,.30,.10,.03,.02]),
}
TIER_BTYPE   = {1:([0,1,2],[.50,.32,.18]), 2:([0,1,2],[.62,.25,.13]), 3:([0,1,2],[.78,.16,.06])}
TIER_QUALITY = {1:([0,1,2],[.15,.45,.40]), 2:([0,1,2],[.25,.52,.23]), 3:([0,1,2],[.45,.45,.10])}
TIER_NOISE   = {1:.045, 2:.055, 3:.065}
TIER_COMPLEX = {1:(.90,1.10), 2:(.92,1.08), 3:(.94,1.06)}

base = os.path.dirname(os.path.abspath(__file__))
cdf  = pd.read_csv(os.path.join(base, "city_rates.csv"))[
    ["city","state","tier","cement","sand","brick","aggregate","steel"]]

print(f"[INFO] Cities loaded: {len(cdf)}  |  "
      + "  ".join(f"T{t}:{len(cdf[cdf.tier==t])}" for t in [1,2,3]))

frames = []
for tier, n in TIER_SAMPLES.items():
    rng = np.random.default_rng(SEED + tier)
    tc  = cdf[cdf["tier"] == tier].reset_index(drop=True)
    idx = rng.integers(0, len(tc), n)

    area   = rng.uniform(*TIER_AREA[tier], n)
    fl_c, fl_p = TIER_FLOORS[tier];  floors = rng.choice(fl_c, n, p=fl_p)
    bt_c, bt_p = TIER_BTYPE[tier];   btype  = rng.choice(bt_c, n, p=bt_p)
    q_c,  q_p  = TIER_QUALITY[tier]; qual   = rng.choice(q_c,  n, p=q_p)
    ta = area * floors

    q_m    = np.where(qual==0, .82,  np.where(qual==1, 1.0, 1.22))
    bt_cem = np.where(btype==0, 1.,  np.where(btype==1, 1.12, 1.20))
    bt_st  = np.where(btype==0, 1.,  np.where(btype==1, 1.25, 1.45))
    bt_ag  = np.where(btype==0, 1.,  np.where(btype==1, 1.10, 1.18))
    ff     = 1.0 + np.clip((floors - 1) * .018, 0, .20)
    lo, hi = TIER_COMPLEX[tier]
    cx     = rng.uniform(lo, hi, n)
    sg     = TIER_NOISE[tier]

    uc = tc["cement"].values[idx].astype(float)
    us = tc["sand"].values[idx].astype(float)
    ub = tc["brick"].values[idx].astype(float)
    ua = tc["aggregate"].values[idx].astype(float)
    ut = tc["steel"].values[idx].astype(float)

    sand_q = np.where(qual==0, .88, np.where(qual==1, 1., 1.15))
    brk_b  = np.where(btype==0, 6.5, np.where(btype==1, 5., 4.))
    brk_q  = np.where(qual==0, .90,  np.where(qual==1, 1., 1.10))
    stl_b  = np.where(btype==0, 4.,  np.where(btype==1, 5., 5.8))
    stl_q  = np.where(qual==0, .88,  np.where(qual==1, 1., 1.18))

    cem = np.clip(ta*.42*q_m*bt_cem*ff*cx + rng.normal(0, ta*sg,    n), 1, None).astype(int)
    san = np.clip(ta*1.45*sand_q*cx        + rng.normal(0, ta*sg*1.1,n), 1, None).astype(int)
    brk = np.clip(ta*brk_b*brk_q*cx       + rng.normal(0, ta*sg*2., n), 0, None).astype(int)
    agg = np.clip(ta*1.20*q_m*bt_ag*ff*cx + rng.normal(0, ta*sg,    n), 1, None).astype(int)
    stl = np.clip(ta*stl_b*stl_q*bt_st*ff*cx + rng.normal(0, ta*sg*1.2,n), 1, None).astype(int)

    cc = cem*uc; cs = san*us; cb = brk*ub; ca = agg*ua; ct = stl*ut
    total = cc + cs + cb + ca + ct

    frames.append(pd.DataFrame({
        "city":           tc["city"].values[idx],
        "state":          tc["state"].values[idx],
        "tier":           tier,
        "area_sqft":      area.round(2),
        "floors":         floors,
        "building_type":  btype,
        "quality":        qual,
        "total_area_sqft":ta.round(2),
        "cement_bags":    cem,
        "sand_cft":       san,
        "bricks":         brk,
        "aggregate_cft":  agg,
        "steel_kg":       stl,
        "price_cement_per_bag": uc.astype(int),
        "price_sand_per_cft":   us.astype(int),
        "price_brick_per_nos":  ub.astype(int),
        "price_agg_per_cft":    ua.astype(int),
        "price_steel_per_kg":   ut.astype(int),
        "cost_cement":    cc.astype(int),
        "cost_sand":      cs.astype(int),
        "cost_bricks":    cb.astype(int),
        "cost_aggregate": ca.astype(int),
        "cost_steel":     ct.astype(int),
        "total_material_cost": total.astype(int),
    }))
    print(f"[GEN] Tier {tier}: {n:,} samples | {tc['city'].nunique()} cities | "
          f"area {area.min():.0f}-{area.max():.0f} sqft")

df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
out = os.path.join(base, "data.csv")
df.to_csv(out, index=False)

print(f"\n[OK] Saved -> {out}")
print(f"     Rows : {len(df):,}  |  Cols : {len(df.columns)}")
print(f"     Cities: {df['city'].nunique()} across Tier 1/2/3")

grp = df.groupby("tier").agg(
    samples=("tier","count"), cities=("city","nunique"),
    avg_area=("total_area_sqft","mean"),
    avg_cost=("total_material_cost","mean")
)
grp["cost_per_sqft"] = (grp["avg_cost"]/grp["avg_area"]).round(0).astype(int)
grp["avg_cost"] = grp["avg_cost"].round(0).astype(int)
print("\n--- Per-Tier Summary ---")
print(grp.to_string())

mask = (df.area_sqft.between(950,1050) & (df.floors==1) &
        (df.building_type==0) & (df.quality==1))
s = df[mask]
print(f"\n--- Benchmark ~1000sqft 1F Residential Standard ({len(s)} samples) ---")
for name, col, lo, hi in [
    ("Cement bags",     "cement_bags",   400, 500),
    ("Sand cft",        "sand_cft",     1200,1800),
    ("Bricks nos",      "bricks",       5000,8000),
    ("Aggregate cft",   "aggregate_cft",1000,1500),
    ("Steel kg",        "steel_kg",     3500,5500),
]:
    m = int(s[col].mean()) if len(s) else 0
    ok = "OK" if lo<=m<=hi else "!"
    print(f"  {name:<18s} {m:>5d}   (expected {lo}-{hi}) [{ok}]")
