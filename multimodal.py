"""Multimodal network: tabular + image embedding + text embedding -> log price.

Run after baseline.py and features.py:   python multimodal.py
Options: --seeds 5 --folds 5 --epochs 300 --quick (fewer seeds/folds for a test run)

Ablation: tabular only / +text / +images / all three, plus the linear baseline.
Everything is evaluated (a) on the SAME hold-out split as baseline.py
(mean +- sd over seeds) and (b) with K-fold CV over all houses.
"""
import argparse, copy, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from common import load, metrics, tab_preprocessor

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--img_pca", type=int, default=64)
ap.add_argument("--txt_pca", type=int, default=32)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
if a.quick:
    a.seeds, a.folds, a.epochs = 2, 3, 80

df = load()
ids = pd.read_csv("data/features/ids.csv").listing_id
IMG = pd.DataFrame(np.load("data/features/img_emb.npy"), index=ids).loc[df.listing_id].values
TXT = pd.DataFrame(np.load("data/features/text_emb.npy"), index=ids).loc[df.listing_id].values
y_all = df.log_price.values
CONFIGS = {"tabular": (False, False), "tabular+text": (True, False),
           "tabular+images": (False, True), "tabular+text+images": (True, True)}


def prep(tr, te):
    """Fit all preprocessing on the training rows only."""
    pre = tab_preprocessor(df).fit(df.iloc[tr])
    out = {"tab": (pre.transform(df.iloc[tr]), pre.transform(df.iloc[te]))}
    for name, M, k in [("img", IMG, a.img_pca), ("txt", TXT, a.txt_pca)]:
        sc = StandardScaler().fit(M[tr])
        pca = PCA(k, random_state=0).fit(sc.transform(M[tr]))
        scale = pca.transform(sc.transform(M[tr]))[:, 0].std()   # keep relative variances
        out[name] = tuple(pca.transform(sc.transform(M[i])) / scale for i in (tr, te))
    return out


class Net(nn.Module):
    def __init__(self, d_tab, d_txt, d_img, use_txt, use_img, p=0.3):
        super().__init__()
        self.use_txt, self.use_img = use_txt, use_img
        self.tab = nn.Sequential(nn.Linear(d_tab, 64), nn.ReLU(), nn.Dropout(p))
        self.txt = nn.Sequential(nn.Dropout(p), nn.Linear(d_txt, 32), nn.ReLU()) if use_txt else None
        self.img = nn.Sequential(nn.Dropout(p), nn.Linear(d_img, 32), nn.ReLU()) if use_img else None
        d = 64 + 32 * (use_txt + use_img)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(p), nn.Linear(64, 1))

    def forward(self, tab, txt, img):
        z = [self.tab(tab)]
        if self.use_txt: z.append(self.txt(txt))
        if self.use_img: z.append(self.img(img))
        return self.head(torch.cat(z, 1)).squeeze(1)


def T(x): return torch.tensor(np.asarray(x), dtype=torch.float32)


def fit_predict(cfg, tr, te, seed, P):
    use_txt, use_img = CONFIGS[cfg]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr)); nval = max(1, int(.1 * len(tr)))
    vi, ti = perm[:nval], perm[nval:]
    mu, sd = y_all[tr].mean(), y_all[tr].std()
    ytr = T((y_all[tr] - mu) / sd)
    X = [T(P["tab"][0]), T(P["txt"][0]), T(P["img"][0])]
    Xte = [T(P["tab"][1]), T(P["txt"][1]), T(P["img"][1])]
    net = Net(X[0].shape[1], X[1].shape[1], X[2].shape[1], use_txt, use_img)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, bad = 1e9, None, 0
    for ep in range(a.epochs):
        net.train(); order = rng.permutation(ti)
        for b in range(0, len(order), 64):
            idx = order[b:b + 64]
            loss = nn.functional.smooth_l1_loss(net(*[x[idx] for x in X]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(*[x[vi] for x in X]), ytr[vi]).item()
        if v < best - 1e-4: best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience: break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(*Xte).numpy() * sd + mu


def agg(runs):
    out = {}
    for scale in ("log", "usd"):
        for m in ("RMSE", "MAE", "R2"):
            v = [r[scale][m] for r in runs]
            out[f"{scale}_{m}_mean"], out[f"{scale}_{m}_sd"] = float(np.mean(v)), float(np.std(v))
    return out


# ---------------- (a) fixed hold-out split ----------------
tr = np.where(df.split == "train")[0]; te = np.where(df.split == "test")[0]
P = prep(tr, te)
lin = LinearRegression().fit(P["tab"][0], y_all[tr])
rows, preds = {}, {}
rows["linear baseline"] = agg([metrics(y_all[te], lin.predict(P["tab"][1]))])
for cfg in CONFIGS:
    ps = [fit_predict(cfg, tr, te, s, P) for s in range(a.seeds)]
    preds[cfg] = np.mean(ps, 0)
    rows[cfg] = agg([metrics(y_all[te], p) for p in ps])
    rows[cfg]["ensemble_log_R2"] = metrics(y_all[te], preds[cfg])["log"]["R2"]
    print(f"[holdout] {cfg:22s} log R2 {rows[cfg]['log_R2_mean']:.3f} +- {rows[cfg]['log_R2_sd']:.3f}"
          f" | USD MAE {rows[cfg]['usd_MAE_mean']:,.0f}")
hold = pd.DataFrame(rows).T
Path("results").mkdir(exist_ok=True)
hold.round(4).to_csv("results/ablation_holdout.csv")

# ---------------- (b) K-fold CV over all houses ----------------
kf = KFold(a.folds, shuffle=True, random_state=42)
cv = {k: [] for k in ["linear baseline", *CONFIGS]}
for f, (tri, tei) in enumerate(kf.split(df)):
    P = prep(tri, tei)
    cv["linear baseline"].append(metrics(y_all[tei], LinearRegression().fit(P["tab"][0], y_all[tri]).predict(P["tab"][1])))
    for cfg in CONFIGS:
        cv[cfg].append(metrics(y_all[tei], fit_predict(cfg, tri, tei, f, P)))
    print(f"[cv] fold {f + 1}/{a.folds} done")
cvt = pd.DataFrame({k: agg(v) for k, v in cv.items()}).T
cvt.round(4).to_csv("results/ablation_cv.csv")
print("\nHold-out (same split as baseline):\n", hold[["log_RMSE_mean", "log_MAE_mean", "log_R2_mean", "log_R2_sd", "usd_MAE_mean", "usd_R2_mean"]].round(3))
print(f"\n{a.folds}-fold CV:\n", cvt[["log_RMSE_mean", "log_MAE_mean", "log_R2_mean", "log_R2_sd", "usd_MAE_mean", "usd_R2_mean"]].round(3))

# ---------------- outputs for the report ----------------
best = "tabular+text+images"
pd.DataFrame({"listing_id": df.listing_id.values[te], "y_log": y_all[te], "pred_log": preds[best]}) \
    .to_csv("results/multimodal_predictions.csv", index=False)
resid = y_all[te] - preds[best]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(preds[best], resid, s=6, alpha=.4); ax[0].axhline(0, c="k"); ax[0].set_title("Residuals vs fitted")
ax[1].hist(resid, bins=40); ax[1].set_title("Residual distribution")
ax[2].scatter(y_all[te], preds[best], s=6, alpha=.4)
lim = [y_all[te].min(), y_all[te].max()]; ax[2].plot(lim, lim, "k--"); ax[2].set_title("Predicted vs actual")
fig.tight_layout(); fig.savefig("results/multimodal_residuals.png", dpi=150)
Path("results/multimodal_metrics.json").write_text(json.dumps({"holdout": rows, "cv": cvt.to_dict("index")}, indent=2))