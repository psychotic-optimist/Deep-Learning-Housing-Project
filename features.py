"""Extract frozen pretrained embeddings (run ONCE):  python features.py

  images -> data/features/img_emb.npy   (mean over up to 3 photos per house)
  text   -> data/features/text_emb.npy  (sentence-transformer of the description)
  order  -> data/features/ids.csv       (listing_id per row, plus has_img flag)

pip install torch torchvision sentence-transformers pillow
Add --model resnet18 for a lighter/faster CNN.  Safe to rerun (skips finished parts).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50", choices=["resnet50", "resnet18"])
ap.add_argument("--per", type=int, default=3)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

OUT = Path("data/features"); OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/houses_clean.csv")
ids = df.listing_id.tolist()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---------------- images ----------------
img_file = OUT / "img_emb.npy"
if img_file.exists() and not a.force:
    print("img_emb.npy exists - skipping images")
else:
    import torchvision
    from torchvision import models
    W = {"resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
         "resnet18": models.ResNet18_Weights.IMAGENET1K_V1}[a.model]
    net = getattr(models, a.model)(weights=W)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    prep = W.transforms()                     # resize 256 / crop 224 / ImageNet normalise

    files = []                                # (row index, path)
    for i, lid in enumerate(ids):
        for k in range(1, a.per + 1):
            p = Path("images") / (f"{lid}.webp" if k == 1 else f"{lid}_{k}.webp")
            if p.exists():
                files.append((i, p))
    print(f"{len(files)} photos for {len(ids)} listings, model={a.model}")

    dim = 2048 if a.model == "resnet50" else 512
    sums = np.zeros((len(ids), dim), dtype=np.float32)
    cnt = np.zeros(len(ids))
    with torch.no_grad():
        for b in range(0, len(files), a.batch):
            chunk, rows = [], []
            for i, p in files[b:b + a.batch]:
                try:
                    chunk.append(prep(Image.open(p).convert("RGB"))); rows.append(i)
                except Exception as e:
                    print("  skip", p.name, e)
            if chunk:
                emb = net(torch.stack(chunk).to(dev)).cpu().numpy()
                for r, e in zip(rows, emb):
                    sums[r] += e; cnt[r] += 1
            if (b // a.batch) % 20 == 0:
                print(f"  {b}/{len(files)}")
    has = cnt > 0
    emb = np.zeros_like(sums); emb[has] = sums[has] / cnt[has, None]
    emb[~has] = emb[has].mean(0)              # houses with no readable photo -> average photo
    np.save(img_file, emb)
    pd.DataFrame({"listing_id": ids, "has_img": has}).to_csv(OUT / "ids.csv", index=False)
    print(f"saved {img_file}  ({(~has).sum()} houses had no photo)")

# ---------------- text ----------------
txt_file = OUT / "text_emb.npy"
if txt_file.exists() and not a.force:
    print("text_emb.npy exists - skipping text")
else:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
    texts = df.description.fillna("").astype(str).tolist()
    emb = st.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(txt_file, emb.astype(np.float32))
    print(f"saved {txt_file}  shape={emb.shape}")
if not (OUT / "ids.csv").exists():
    pd.DataFrame({"listing_id": ids, "has_img": True}).to_csv(OUT / "ids.csv", index=False)