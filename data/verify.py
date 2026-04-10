import pandas as pd, os
df = pd.read_csv(r"d:\2026\apex-gpt\data\data.csv")
print(f"File: data.csv")
print(f"Rows: {len(df):,}  |  Cols: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")
print()
t = df.groupby("tier").agg(samples=("tier","count"), cities=("city","nunique"))
print("Tier Breakdown:")
print(t.to_string())
print()
print("Cities per tier:")
for tier in [1,2,3]:
    cities = sorted(df[df["tier"]==tier]["city"].unique())
    print(f"  Tier {tier} ({len(cities)} cities): {cities}")
sz = os.path.getsize(r"d:\2026\apex-gpt\data\data.csv") / 1024
print(f"\nFile size: {sz:.1f} KB")
