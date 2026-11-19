"""Linear-regression baseline on log price (run: python baseline.py)

Saves the train/test split to data/split.csv so the deep model can be
evaluated on EXACTLY the same hold-out houses.
"""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
df = pd.read_csv("data/houses_clean.csv")

num = ["bedrooms", "bathrooms", "log_floor", "log_land", "latitude", "longitude",
       "gps_exact", "n_amenities", "desc_len"] + [c for c in df if c.startswith("am_")]
cat = ["property_type", "region", "suburb"]

train, test = train_test_split(df, test_size=0.2, random_state=SEED)
Path("data").mkdir(exist_ok=True)
pd.concat([train.assign(split="train"), test.assign(split="test")])[["listing_id", "split"]] \
    .to_csv("data/split.csv", index=False)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5), cat),
])
model = Pipeline([("pre", pre), ("lr", LinearRegression())])

# 5-fold CV on the training set (R2 on log price)
cv = cross_val_score(model, train[num + cat], train.log_price,
                     cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
model.fit(train[num + cat], train.log_price)
pred_log = model.predict(test[num + cat])
pred_usd, y_usd = np.exp(pred_log), test.price.values

def metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p))}

res = {"n_train": len(train), "n_test": len(test),
       "cv_r2_log_mean": float(cv.mean()), "cv_r2_log_std": float(cv.std()),
       "test_log_scale": metrics(test.log_price, pred_log),
       "test_usd_scale": metrics(y_usd, pred_usd)}
Path("results").mkdir(exist_ok=True)
Path("results/baseline_metrics.json").write_text(json.dumps(res, indent=2))
pd.DataFrame({"listing_id": test.listing_id.values, "y_log": test.log_price.values,
              "pred_log": pred_log}).to_csv("results/baseline_predictions.csv", index=False)
print(json.dumps(res, indent=2))

# residual diagnostics
resid = test.log_price.values - pred_log
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(pred_log, resid, s=6, alpha=.4); ax[0].axhline(0, c="k")
ax[0].set_xlabel("predicted log price"); ax[0].set_ylabel("residual"); ax[0].set_title("Residuals vs fitted")
ax[1].hist(resid, bins=40); ax[1].set_title("Residual distribution")
ax[2].scatter(test.log_price, pred_log, s=6, alpha=.4)
lim = [test.log_price.min(), test.log_price.max()]; ax[2].plot(lim, lim, "k--")
ax[2].set_xlabel("actual log price"); ax[2].set_ylabel("predicted"); ax[2].set_title("Predicted vs actual")
fig.tight_layout(); fig.savefig("results/baseline_residuals.png", dpi=150)