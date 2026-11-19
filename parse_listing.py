"""parse_listing(html) -> dict for one property.co.zw listing page.

Used by Crawler.py (details stage). Importing this file has NO side effects.
Reads the schema.org JSON-LD block first, then falls back to <meta> tags and
visible HTML. Raises ValueError if the page doesn't look like a listing.

To debug on one page:  python parse_listing.py path/to/saved_listing.html
"""
import json
import re
import sys

from bs4 import BeautifulSoup

try:                      # lxml is several times faster; falls back to html.parser
    import lxml  # noqa: F401
    PARSER = "lxml"
except ImportError:
    PARSER = "html.parser"


def _walk(obj):
    """Yield every dict inside a nested JSON structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _meta(soup, **attrs):
    tag = soup.find("meta", attrs=attrs)
    return tag.get("content") if tag else None


def _float(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_listing(html):
    soup = BeautifulSoup(html, PARSER)

    # --- JSON-LD: pick the block that has price/about info
    ld = {}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "", strict=False)
        except json.JSONDecodeError:
            continue
        for d in _walk(data):
            if "offers" in d or "about" in d:
                ld = d
                break
        if ld:
            break

    offers = ld.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    about = ld.get("about") or {}
    addr = about.get("address") or {}
    geo = about.get("geo") or ld.get("geo") or {}

    canon = soup.find("link", rel="canonical")
    url = ld.get("url") or (canon.get("href") if canon else None) or _meta(soup, property="og:url")

    listing_id = ld.get("identifier")
    if not listing_id and url:
        m = re.search(r"(\d+)/?$", url)
        listing_id = m.group(1) if m else None

    price = _float(offers.get("price"))
    if not listing_id and price is None:
        raise ValueError("no listing id or price found - not a listing page?")

    # --- coordinates: exact GPS if the agent set it, else suburb centroid (flagged)
    lat, lon = _float(geo.get("latitude")), _float(geo.get("longitude"))
    gps_exact = lat is not None
    if not gps_exact:
        m = re.search(r'\\?"Lat\\?":(-?\d+\.\d+),\\?"Lng\\?":(-?\d+\.\d+)', html)
        if m:
            lat, lon, gps_exact = float(m.group(1)), float(m.group(2)), True
    sub_lat = sub_lon = None
    m = re.search(r'SuburbLat\\?":(-?\d+\.\d+),\\?"SuburbLng\\?":(-?\d+\.\d+)', html)
    if m:
        sub_lat, sub_lon = float(m.group(1)), float(m.group(2))

    # --- sizes from the visible bed/bath/area block (JSON-LD floorSize is unreliable)
    floor = land = None
    for div in soup.select("div.bed-bath-area-container div.area"):
        txt = div.get_text(" ", strip=True)
        m = re.search(r"([\d,\.]+)\s*m", txt)
        if not m:
            continue
        val = _float(m.group(1))
        if div.find("img", alt="Land area"):
            land = val
        else:
            floor = val
    if floor is not None and floor < 10:      # site uses 1 m2 as a placeholder
        floor = None
    if land is not None and land < 10:
        land = None

    # --- images: JSON-LD, then og:image, de-duplicated in order
    imgs = about.get("image") or ld.get("image") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    og = _meta(soup, property="og:image")
    if og:
        imgs = list(imgs) + [og]
    seen, image_urls = set(), []
    for u in imgs:
        if isinstance(u, dict):
            u = u.get("url")
        if u and u not in seen:
            seen.add(u)
            image_urls.append(u)

    # --- amenities: names listed under the "Amenities" heading
    amenities = []
    for h in soup.find_all(["h2", "h3"]):
        if h.get_text(strip=True) == "Amenities":
            for img in h.parent.find_all("img", alt=True):
                t = img["alt"].strip()
                if t and t not in amenities:
                    amenities.append(t)
            break

    # full description is in div.article; JSON-LD only holds the first ~160 chars
    art = soup.select_one("div.article")
    description = (art.get_text(" ", strip=True) if art else "") \
        or ld.get("description") or _meta(soup, property="og:description") or ""
    description = re.sub(r"\s+", " ", description)

    return {
        "listing_id": listing_id,
        "url": url,
        "date_posted": ld.get("datePosted"),
        "price": price,
        "currency": offers.get("priceCurrency"),
        "agency": (offers.get("seller") or {}).get("name"),
        "bedrooms": about.get("numberOfBedrooms"),
        "bathrooms": about.get("numberOfBathroomsTotal"),
        "floor_size_m2": floor,
        "land_size_m2": land,
        "suburb": addr.get("addressLocality"),
        "region": addr.get("addressRegion"),
        "latitude": lat if gps_exact else sub_lat,
        "longitude": lon if gps_exact else sub_lon,
        "gps_exact": gps_exact,
        "amenities": "; ".join(amenities),
        "description": description.strip(),
        "image_urls": image_urls,
    }


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        print(json.dumps(parse_listing(f.read()), indent=2, ensure_ascii=False))