"""Shared helpers for the deep-model scripts (baseline.py stays standalone)."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT = ["property_type", "region", "suburb"]


def load():
    df = pd.read_csv("data/houses_clean.csv")
    split = pd.read_csv("data/split.csv").set_index("listing_id").split
    df["split"] = df.listing_id.map(split)
    if df.split.isna().any():
        raise SystemExit("data/split.csv does not match houses_clean.csv - rerun baseline.py")
    return df


def num_cols(df):
    return ["bedrooms", "bathrooms", "log_floor", "log_land", "latitude", "longitude",
            "gps_exact", "n_amenities", "desc_len"] + [c for c in df if c.startswith("am_")]


def tab_preprocessor(df):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num_cols(df)),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                              sparse_output=False), CAT),
    ])


def metrics(y_log, p_log):
    y_usd, p_usd = np.exp(y_log), np.exp(p_log)
    f = lambda y, p: {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
                      "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p))}
    return {"log": f(y_log, p_log), "usd": f(y_usd, p_usd)}