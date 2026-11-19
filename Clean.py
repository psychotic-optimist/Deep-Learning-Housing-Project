"""Clean data/houses_raw.csv -> data/houses_clean.csv  (run: python clean.py)

Steps: dedupe, one currency (USD), drop impossible/outlier prices, fix
implausible sizes, build model features (amenity flags, log price, etc).
Missing numeric values are LEFT as NaN here; they are imputed inside the
model pipelines so that no information leaks from the test set.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW, OUT = Path("data/houses_raw.csv"), Path("data/houses_clean.csv")
AMENITY_MIN_SHARE = 0.03          # keep amenity flags present in >=3% of houses

df = pd.read_csv(RAW)
log = [f"raw rows: {len(df)}"]

def step(msg):
    log.append(f"{msg}: {len(df)} rows left")

# 1. duplicates: same ID, or identical (description, price, suburb)
df = df.drop_duplicates("listing_id")
step("after dropping duplicate listing_id")
df = df.drop_duplicates(["description", "price", "suburb"])
step("after dropping duplicate description+price+suburb")

# 2. currency: blank -> USD (site is USD-dominated); anything else is dropped
df["currency"] = df["currency"].fillna("USD").str.upper()
n_other = (df.currency != "USD").sum()
df = df[df.currency == "USD"]
step(f"after dropping {n_other} non-USD listings")

# 3. prices: must be positive; remove extreme outliers on the log scale
df = df[df.price > 0].copy()
df["log_price"] = np.log(df.price)
med = df.log_price.median()
mad = (df.log_price - med).abs().median() * 1.4826
z = (df.log_price - med) / mad
n_out = (z.abs() > 4).sum()
df = df[z.abs() <= 4].copy()
step(f"after removing {n_out} price outliers (|robust z|>4 on log price)")

# 4. implausible values -> missing (not dropped)
df.loc[df.bedrooms > 20, "bedrooms"] = np.nan
df.loc[df.bedrooms <= 0, "bedrooms"] = np.nan
df.loc[df.bathrooms > 20, "bathrooms"] = np.nan
df.loc[(df.floor_size_m2 < 10) | (df.floor_size_m2 > 5000), "floor_size_m2"] = np.nan
df.loc[(df.land_size_m2 < 10) | (df.land_size_m2 > 200000), "land_size_m2"] = np.nan

# 5. text cleanup
for c in ["suburb", "region", "property_type", "agency"]:
    df[c] = df[c].fillna("unknown").astype(str).str.strip().str.title()
df["description"] = df["description"].fillna("").astype(str)

# 6. engineered features
df["n_amenities"] = df.amenities.fillna("").apply(lambda s: len([a for a in s.split(";") if a.strip()]))
df["desc_len"] = df.description.str.split().str.len()
df["log_land"] = np.log1p(df.land_size_m2)
df["log_floor"] = np.log1p(df.floor_size_m2)
df["gps_exact"] = df.gps_exact.astype(str).str.lower().eq("true").astype(int)

am = df.amenities.fillna("").apply(
    lambda s: {a.strip().lower() for a in s.split(";") if a.strip()})
counts = pd.Series([a for s in am for a in s]).value_counts()
keep = counts[counts >= AMENITY_MIN_SHARE * len(df)].index
for a in keep:
    col = "am_" + "".join(ch if ch.isalnum() else "_" for ch in a)
    df[col] = am.apply(lambda s, a=a: int(a in s))
log.append(f"amenity flags kept: {len(keep)} -> {list(keep)}")

df.to_csv(OUT, index=False)
log.append(f"saved {OUT} with {len(df)} rows, {df.shape[1]} columns")
print("\n".join(log))
print("\nMissing share per column:")
print(df.isna().mean().round(3)[lambda s: s > 0])
Path("data/cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")