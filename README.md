# Deep-Learning-Housing-Project
# Zimbabwe House Price Modelling

HDSC405 Deep Learning, University of Zimbabwe: mini research project.

**Research question:** Which features explain the asking prices of houses for sale in Zimbabwe, and how accurately can a model predict them?

Author: _Tanyaradzwa Mbonga_ (_R236009C_)
Lecturer: Mr. P. Watambwa

---

## 1. Overview

This repository contains the full pipeline for the project:

1. **Collect** house listings (structured data, descriptions, photos) from [property.co.zw](https://www.property.co.zw/houses-for-sale).
2. **Clean and explore** the data.
3. **Model** asking prices: a linear regression baseline and a multimodal deep network (tabular features + house photos via a CNN + description text via embeddings/transformers).
4. **Evaluate** on held-out data (RMSE, MAE, R²), check residuals and discuss limitations.

> Status: data collection is implemented. Cleaning, modelling and evaluation sections are filled in below as they are completed.

## 2. Repository structure

```
.
├── Crawler.py            # 4-stage collection pipeline (index, details, images, build)
├── parse_listing.py      # parses one listing page into a dict
├── requirements.txt
├── data/
│   ├── listings_index.csv    # one row per listing found on results pages
│   ├── details.jsonl         # one JSON record per listing page
│   ├── houses_raw.csv        # final raw dataset (joined index + details)
│   └── raw/                  # saved HTML of every page fetched (audit trail)
│       ├── pages/
│       └── listings/
├── images/               # photos; file names start with the listing ID
├── notebooks/ or src/    # cleaning, EDA and modelling code (to be added)
└── report/               # short report (max 6 pages)
```

## 3. Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

`requirements.txt` (data collection):

```
requests
beautifulsoup4
```

Modelling dependencies (e.g. pandas, scikit-learn, torch, transformers) will be added as that code is completed.

## 4. How to run the data collection

Run the stages in order. Each stage can be stopped (Ctrl+C) and restarted safely: finished work is not repeated.

```bash
python Crawler.py index --passes 1      # walk the results pages -> data/listings_index.csv
python Crawler.py details               # fetch each listing page -> data/details.jsonl
python Crawler.py images --per 3        # download up to 3 photos per listing -> images/
python Crawler.py build                 # join everything -> data/houses_raw.csv
```

Useful options for a quick test:

```bash
python Crawler.py index --max-pages 2
python Crawler.py details --limit 5
```

To rerun the whole pipeline from scratch, delete the `data/` and `images/` folders first.

### Responsible collection

- The crawler reads `robots.txt` first and stops if the paths are disallowed.
- It pauses 2 to 3.5 seconds between page requests (0.5 to 1.2 s for image files).
- It backs off on HTTP 429/5xx errors and stops after three consecutive 403 responses.
- Its User-Agent identifies it as a university coursework project and includes a contact email.
- Every record stores `collected_at` (UTC timestamp). Date of collection: _FILL IN START AND END DATES_.
- Raw HTML is saved in `data/raw/` so the data can be re-parsed without re-downloading.
- The data is used only for this coursework project and is not redistributed commercially.

## 5. Data dictionary (`data/houses_raw.csv`)

One row per listing.

| Column | Description |
|---|---|
| `listing_id` | Unique ID from the site (e.g. `KEA259793`). Photo files in `images/` start with this ID (`<id>.webp`, `<id>_2.webp`, ...). |
| `url` | Listing page URL. |
| `property_type` | Property type from the URL slug (e.g. `houses`). |
| `date_posted` | Date the listing was posted. |
| `collected_at` | UTC timestamp when the listing page was collected. |
| `price` | Asking price (numeric). |
| `currency` | Currency of `price` (e.g. USD). |
| `agency` | Selling agency. |
| `bedrooms` | Number of bedrooms. |
| `bathrooms` | Number of bathrooms (missing if not stated). |
| `floor_size_m2` | Building floor area in m². Missing if not stated; the site's placeholder value of 1 m² is treated as missing. |
| `land_size_m2` | Stand/land area in m². Missing if not stated. |
| `suburb` | Suburb name. |
| `region` | Area/region (e.g. Harare South). |
| `latitude`, `longitude` | Location. Exact if the agent supplied GPS, otherwise the suburb centre (see `gps_exact`). |
| `gps_exact` | `True` if coordinates are the property's own, `False` if the suburb centre was used. |
| `amenities` | Amenities listed on the page, separated by `;` (e.g. `Garage; Water Tank; Walled`). |
| `description` | Free-text description written by the agent. |
| `has_image` | Whether the first photo was downloaded. |

The cleaned dataset used for modelling will add: price in a single currency (`price_usd`), `log_price`, and any engineered features (documented here once final).

## 6. Data cleaning and exploration

_To be completed:_ duplicate removal, currency conversion to USD, missing-value handling, outlier treatment, summary tables and plots.

## 7. Models

_To be completed._

- **Baseline:** linear regression on tabular features (target: log price).
- **Multimodal network:** tabular features + CNN image branch + text embedding branch, fused to predict log price.
- Justification of architecture, features and transformations will be given here and in the report.

## 8. Evaluatio

Hold-out set / cross-validation; RMSE, MAE and R²; residual diagnostics; model comparison.

## 9. Limitations



- Asking prices are not sale prices.
- The site lists a limited number of houses, so the sample is limited to what is advertised online and is skewed towards urban areas and agencies that use the site.
- Some fields (bathrooms, floor size, exact GPS) are missing for many listings.

## 10. Reproducing the results

1. Install requirements (section 3).
2. Run the collection stages (section 4), or use the `data/` and `images/` folders included in this repository.
3. Run the cleaning and modelling code (_add exact commands/notebook names when finished_).

## 11. Report

The short report (max 6 pages) is in `report/`.
