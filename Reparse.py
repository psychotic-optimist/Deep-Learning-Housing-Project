
import json
from pathlib import Path
from parse_listing import parse_listing

old = {}
p = Path("data/details.jsonl")
if p.exists():
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line); old[r["listing_id"]] = r.get("collected_at")

out, bad = [], 0
for f in sorted(Path("data/raw/listings").glob("*.html")):
    try:
        rec = parse_listing(f.read_text(encoding="utf-8"))
    except ValueError as e:
        bad += 1; print("skip", f.name, e); continue
    rec["collected_at"] = old.get(rec["listing_id"])
    out.append(rec)
if p.exists():
    p.replace("data/details_backup.jsonl")
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
print(f"re-parsed {len(out)} listings ({bad} skipped) -> {p}")