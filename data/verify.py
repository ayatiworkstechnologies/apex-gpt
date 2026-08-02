import pandas as pd, os
DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
df = pd.read_csv(DATA_CSV)
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
sz = os.path.getsize(DATA_CSV) / 1024
print(f"\nFile size: {sz:.1f} KB")
