"""Summary tables and plots -> report/figures/  (run: python eda.py)"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/houses_clean.csv")
fig = Path("report/figures"); fig.mkdir(parents=True, exist_ok=True)

num = ["price", "bedrooms", "bathrooms", "floor_size_m2", "land_size_m2", "n_amenities", "desc_len"]
summary = df[num].describe().T.round(1)
summary.to_csv(fig / "summary_numeric.csv"); print(summary, "\n")
top = (df.groupby("region").price.agg(["count", "median"]).sort_values("count", ascending=False).head(15))
top.to_csv(fig / "price_by_region.csv"); print(top, "\n")

fig1, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(df.price, bins=50); ax[0].set_title("Asking price (USD)")
ax[1].hist(df.log_price, bins=50); ax[1].set_title("log(price)")
fig1.tight_layout(); fig1.savefig(fig / "price_hist.png", dpi=150)

regions = top.index[:10]
d = df[df.region.isin(regions)]
fig2, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([d[d.region == r].log_price for r in regions])
ax.set_xticklabels(regions)
ax.set_title("log(price) by region (10 most common)"); plt.xticks(rotation=45, ha="right")
fig2.tight_layout(); fig2.savefig(fig / "price_by_region.png", dpi=150)

fig3, ax = plt.subplots(figsize=(5, 4))
d = df.dropna(subset=["floor_size_m2"])
ax.scatter(np.log(d.floor_size_m2), d.log_price, s=6, alpha=.4)
ax.set_xlabel("log floor size"); ax.set_ylabel("log price")
fig3.tight_layout(); fig3.savefig(fig / "price_vs_floor.png", dpi=150)

c = df[["log_price", "bedrooms", "bathrooms", "log_floor", "log_land", "n_amenities", "desc_len"]].corr()
c.round(2).to_csv(fig / "correlations.csv"); print(c.round(2))
print("\nsaved to", fig)