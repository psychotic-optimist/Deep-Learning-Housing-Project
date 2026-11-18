"""Resumable collection pipeline for property.co.zw /houses-for-sale.

Stages (run in order, each can be stopped and restarted safely):
  python crawler.py index    --passes 2   # walk results pages -> data/listings_index.csv
  python crawler.py details               # fetch each listing page -> data/details.jsonl
  python crawler.py images   --per 3      # download photos -> images/<listing_id>.webp
  python crawler.py build                 # join everything -> data/houses_raw.csv

Be polite: it checks robots.txt, pauses 2-3.5 s between requests, backs off on
429/5xx and stops on repeated 403. Put your own contact email in USER_AGENT.
"""
import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from parse_listing import parse_listing

BASE = "https://www.property.co.zw"
START = BASE + "/houses-for-sale"
USER_AGENT = "Mozilla/5.0 (Santa Clause :) - UZ coursework project; contact: tmbonga25@gmail.com)"
PAGE_SIZE = 20
DELAY = (2.0, 3.5)  # seconds between page requests

DATA = Path("data")
RAW_PAGES = DATA / "raw" / "pages"
RAW_LISTINGS = DATA / "raw" / "listings"
IMAGES = Path("images")
INDEX_CSV = DATA / "listings_index.csv"
DETAILS = DATA / "details.jsonl"
OUT_CSV = DATA / "houses_raw.csv"

INDEX_FIELDS = [
    "listing_id", "url", "property_type", "date_posted", "price", "currency",
    "agency", "bedrooms", "bathrooms", "floor_size_m2", "land_size_m2",
    "suburb", "region", "n_images", "snippet", "seen_on_page", "collected_at",
]

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en"})
_403_count = 0


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def polite_get(url, delay=DELAY, tries=4):
    """GET with jitter, backoff on 429/5xx, and a hard stop on repeated 403."""
    global _403_count
    for attempt in range(tries):
        time.sleep(random.uniform(*delay))
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"  ! network error ({e}); retry {attempt + 1}/{tries}")
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            _403_count = 0
            return r
        if r.status_code == 403:
            _403_count += 1
            if _403_count >= 3:
                sys.exit("Blocked (403) three times in a row - stopping. Wait and try later.")
        if r.status_code in (403, 429) or r.status_code >= 500:
            wait = 30 * (attempt + 1)
            print(f"  ! HTTP {r.status_code} for {url}; sleeping {wait}s")
            time.sleep(wait)
            continue
        print(f"  ! HTTP {r.status_code} for {url}; skipping")
        return None
    return None


def check_robots(sample_urls):
    """Abort if robots.txt disallows what we are about to fetch."""
    from urllib import robotparser

    try:
        txt = session.get(BASE + "/robots.txt", timeout=20).text
    except requests.RequestException:
        print("Could not read robots.txt - read it yourself before crawling:", BASE + "/robots.txt")
        return
    rp = robotparser.RobotFileParser()
    rp.parse(txt.splitlines())
    for u in sample_urls:
        if not rp.can_fetch(USER_AGENT, u):
            sys.exit(f"robots.txt disallows {u} - not crawling. Ask your lecturer.")
    print("robots.txt allows the paths we need.")


# ----------------------------------------------------------------- stage 1
def num(text):
    m = re.search(r"\d[\d,\.]*", text or "")
    return float(m.group(0).replace(",", "")) if m else None


def parse_results_page(html):
    """Return (total_results, [row dicts]) for one results page.

    Every card embeds a schema.org RealEstateListing JSON-LD block, so we
    read price/beds/baths/floor size/suburb from that, and land size from the
    visible <span class="land-size"> (it is not in the JSON-LD).
    """
    soup = BeautifulSoup(html, "html.parser")
    total = None
    desc = soup.select_one("div.result-description")
    m = re.search(r"of\s+([\d,]+)\s+properties", desc.get_text(" ", strip=True) if desc else "")
    if m:
        total = int(m.group(1).replace(",", ""))

    rows = []
    for card in soup.select("div.ResultCardItem"):
        tag = card.find("script", type="application/ld+json")
        if not tag or not tag.string:
            continue
        try:
            ld = json.loads(tag.string, strict=False)
        except json.JSONDecodeError:
            continue
        offers, about = ld.get("offers", {}), ld.get("about", {})
        addr = about.get("address", {})
        land = card.select_one("span.land-size")
        url = ld.get("url", "")
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        rows.append({
            "listing_id": ld.get("identifier"),
            "url": url,
            "property_type": slug.split("-")[0] if slug else None,
            "date_posted": ld.get("datePosted"),
            "price": float(offers["price"]) if offers.get("price") else None,
            "currency": offers.get("priceCurrency"),
            "agency": (offers.get("seller") or {}).get("name"),
            "bedrooms": about.get("numberOfBedrooms"),
            "bathrooms": about.get("numberOfBathroomsTotal"),
            "floor_size_m2": (about.get("floorSize") or {}).get("value"),
            "land_size_m2": num(land.get_text(" ", strip=True)) if land else None,
            "suburb": addr.get("addressLocality"),
            "region": addr.get("addressRegion"),
            "n_images": len(about.get("image") or []),
            "snippet": (ld.get("description") or "")[:300],
        })
    return total, rows


def load_seen():
    if not INDEX_CSV.exists():
        return set()
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        return {r["listing_id"] for r in csv.DictReader(f)}


def crawl_index(passes=1, sort="listeddesc", max_pages=None):
    """Walk the results pages. The site's default order is random, so we ask
    for a fixed sort and also allow several passes: the index is append-only
    and de-duplicated, so extra passes only add listings missed earlier."""
    DATA.mkdir(exist_ok=True)
    RAW_PAGES.mkdir(parents=True, exist_ok=True)
    check_robots([START + "?page=2"])
    seen = load_seen()
    new_file = not INDEX_CSV.exists()
    with open(INDEX_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new_file:
            w.writeheader()
        for p in range(1, passes + 1):
            page, last = 1, None
            while True:
                url = f"{START}?sortby={sort}" + (f"&page={page}" if page > 1 else "")
                r = polite_get(url)
                if r is None:
                    print(f"pass {p} page {page}: failed, moving on")
                    page += 1
                    continue
                (RAW_PAGES / f"pass{p}_page{page:03d}.html").write_text(r.text, encoding="utf-8")
                total, rows = parse_results_page(r.text)
                if total:
                    last = math.ceil(total / PAGE_SIZE)
                added = 0
                for row in rows:
                    if row["listing_id"] and row["listing_id"] not in seen:
                        seen.add(row["listing_id"])
                        row.update(seen_on_page=page, collected_at=now_iso())
                        w.writerow(row)
                        added += 1
                f.flush()
                print(f"pass {p} page {page}/{last}: {len(rows)} cards, {added} new, {len(seen)} total")
                if not rows or (last and page >= last) or (max_pages and page >= max_pages):
                    break
                page += 1
    print(f"\nIndex has {len(seen)} unique listings (site reports {total}).")


# ----------------------------------------------------------------- stage 2
def read_index():
    with open(INDEX_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_details():
    if not DETAILS.exists():
        return {}
    out = {}
    with open(DETAILS, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["listing_id"]] = rec
    return out


def crawl_details(limit=None):
    RAW_LISTINGS.mkdir(parents=True, exist_ok=True)
    done = set(read_details())
    todo = [r for r in read_index() if r["listing_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} done, {len(todo)} to fetch")
    with open(DETAILS, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            r = polite_get(row["url"])
            if r is None:
                continue
            (RAW_LISTINGS / f"{row['listing_id']}.html").write_text(r.text, encoding="utf-8")
            try:
                rec = parse_listing(r.text)
            except ValueError as e:
                print(f"  ! {row['listing_id']}: {e}")
                continue
            rec["collected_at"] = now_iso()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")


# ----------------------------------------------------------------- stage 3
def download_images(per=3):
    IMAGES.mkdir(exist_ok=True)
    details = read_details()
    for n, (lid, rec) in enumerate(details.items(), 1):
        for i, url in enumerate((rec.get("image_urls") or [])[:per], 1):
            path = IMAGES / (f"{lid}.webp" if i == 1 else f"{lid}_{i}.webp")
            if path.exists():
                continue
            r = polite_get(url, delay=(0.5, 1.2))
            if r is not None:
                path.write_bytes(r.content)
        if n % 50 == 0:
            print(f"  images: {n}/{len(details)} listings")


# ----------------------------------------------------------------- stage 4
def build_csv():
    index = {r["listing_id"]: r for r in read_index()}
    details = read_details()
    cols = [
        "listing_id", "url", "property_type", "date_posted", "collected_at", "price",
        "currency", "agency", "bedrooms", "bathrooms", "floor_size_m2", "land_size_m2",
        "suburb", "region", "latitude", "longitude", "gps_exact", "amenities", "description", "has_image",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for lid, rec in details.items():
            row = {c: rec.get(c) for c in cols}
            row["property_type"] = index.get(lid, {}).get("property_type")
            row["has_image"] = (IMAGES / f"{lid}.webp").exists()
            w.writerow(row)
    print(f"Wrote {len(details)} rows to {OUT_CSV}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["index", "details", "images", "build"])
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--sort", default="listeddesc")
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--limit", type=int, help="details: only fetch this many (for a test run)")
    ap.add_argument("--per", type=int, default=3, help="images per listing")
    a = ap.parse_args()
    if a.stage == "index":
        crawl_index(a.passes, a.sort, a.max_pages)
    elif a.stage == "details":
        crawl_details(a.limit)
    elif a.stage == "images":
        download_images(a.per)
    else:
        build_csv()