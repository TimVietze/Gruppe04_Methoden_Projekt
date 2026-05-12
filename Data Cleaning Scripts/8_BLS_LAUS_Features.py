"""
Phase V2.3 — BLS LAUS Local Unemployment Feature Pipeline
==========================================================

PURPOSE
-------
Download BLS Local Area Unemployment Statistics (LAUS) for all 855 modeling
CBSAs, aggregate to annual frequency, and left-join into the v2 macro modeling
table to create a v2_macro_laus input file.  This is a data-preparation step
only; no model is trained here.

DATA SOURCES
------------
Census CBSA Delineation File  (local, read-only)
  Path   : Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv
  Used to classify each CBSA as Metropolitan or Micropolitan and to derive the
  principal state FIPS code (state with most counties in multi-state CBSAs).

BLS LAUS API v2 (registration key required)
  URL      : https://api.bls.gov/publicAPI/v2/timeseries/data/
  Key      : Set env var BLS_API_KEY before running. Register free at
             https://data.bls.gov/registrationEngine/
  Grain    : Monthly, CBSA level
  Windows  : 2009–2018 and 2019–2025 (API limited to 10-year windows per request)
  Batch    : 50 series per request (key allows up to 50)
  Series format:
    Metropolitan MSA : LAUMT + state_fips(2) + cbsa(5) + 000000 + 03
    Micropolitan μSA : LAUMC + state_fips(2) + cbsa(5) + 000000 + 03
  All 855 series confirmed accessible during pre-implementation exploration.

FRED UNRATE
  URL      : https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE
  Grain    : Monthly, national
  Aggregation : Annual mean

FEATURES PRODUCED
-----------------
unemployment_rate        Annual mean of monthly LAUS values for year t
unemployment_yoy         YoY pp change in unemployment_rate (NaN for first year)
unemployment_vs_national unemployment_rate minus national UNRATE annual mean

LEAKAGE LOGIC
-------------
Unemployment for year t is fully published before year t+1 begins.  The
modeling target target_growth_1y covers Dec(t) to Dec(t+1).  Features for
year t do not use any t+1 information.  No shift is required.

INPUTS  (read-only)
-------------------
  Methoden Data/Modeling Data/Annual_Modeling_Table_v2_macro.csv
  Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv

OUTPUTS  (new files only — no existing file is modified)
---------------------------------------------------------
  Methoden Data/Economic Data/bls_laus_cbsa_monthly_raw.csv
  Methoden Data/Economic Data/bls_laus_cbsa_annual.csv
  Methoden Data/Modeling Data/Annual_Modeling_Table_v2_macro_laus.csv
"""

import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

# ── module-level leakage guards ────────────────────────────────────────────────
NEW_COLS = ["unemployment_rate", "unemployment_yoy", "unemployment_vs_national"]
assert all("next" not in c.lower() for c in NEW_COLS)
assert "cpi_avg" not in NEW_COLS

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
ECON_DIR     = ROOT / "Methoden Data" / "Economic Data"
ECON_DIR.mkdir(parents=True, exist_ok=True)

V2_MAC_TABLE   = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table_v2_macro.csv"
LAUS_TABLE     = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table_v2_macro_laus.csv"
MONTHLY_OUT    = ECON_DIR / "bls_laus_cbsa_monthly_raw.csv"
ANNUAL_OUT     = ECON_DIR / "bls_laus_cbsa_annual.csv"
CATALOG_CACHE  = ECON_DIR / "bls_la_series_catalog.tsv"   # cached flat file, never committed

BLS_API_URL  = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
UNRATE_URL   = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

# BLS API key — required. Register free at https://data.bls.gov/registrationEngine/
# Set via: export BLS_API_KEY="your_key_here"  before running this script.
_raw_key = os.environ.get("BLS_API_KEY", "").strip()
if not _raw_key:
    raise RuntimeError(
        "BLS_API_KEY environment variable is not set.\n"
        "  Register free at https://data.bls.gov/registrationEngine/\n"
        "  Then run:  export BLS_API_KEY=your_key && python3 "
        "'Data Cleaning Scripts/8_BLS_LAUS_Features.py'"
    )
BLS_API_KEY = _raw_key
BATCH_SIZE  = 50   # key allows up to 50 series per request
print(f"BLS API key set — batch size: {BATCH_SIZE}")

YEAR_WINDOWS = [("2009", "2018"), ("2019", "2025")]
MAX_RETRIES  = 4

SEP = "─" * 64


# ── helpers ────────────────────────────────────────────────────────────────────
def _fetch_catalog(retries: int = MAX_RETRIES) -> str:
    """Download la.series from download.bls.gov and return as plain text.

    This is the BLS flat-file server — NOT the time-series API.  No API key
    is required or sent.  Handles gzip-compressed responses automatically.
    """
    url = "https://download.bls.gov/pub/time.series/la/la.series"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "*/*",
    }
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            try:
                return gzip.decompress(raw).decode("utf-8", errors="replace")
            except (gzip.BadGzipFile, OSError):
                return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download BLS la.series catalog after {retries} attempts: {exc}\n"
                    "  Check network access to download.bls.gov"
                )
            wait = 2 ** attempt
            print(f"    Catalog retry {attempt} after {wait}s — {exc}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _post_bls(series_ids: list[str], start: str, end: str,
              retries: int = MAX_RETRIES) -> dict:
    """POST a BLS API v2 request with exponential-backoff retry.

    Raises RuntimeError immediately on daily rate-limit responses so the
    caller can surface a clear error rather than silently collecting partial data.
    """
    body: dict = {"seriesid": series_ids, "startyear": start, "endyear": end}
    if BLS_API_KEY:
        body["registrationkey"] = BLS_API_KEY
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent":   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    }
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(BLS_API_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.loads(r.read())
            # Detect daily rate limit — fail immediately, not silently
            if resp.get("status") == "REQUEST_NOT_PROCESSED":
                msgs = resp.get("message", [])
                if any("daily threshold" in m for m in msgs):
                    raise RuntimeError(
                        "BLS API daily rate limit reached.\n"
                        "  Without a key: 500 requests/day.\n"
                        "  Register free at https://data.bls.gov/registrationEngine/\n"
                        "  Then re-run: export BLS_API_KEY=your_key && python3 "
                        "'Data Cleaning Scripts/8_BLS_LAUS_Features.py'"
                    )
            return resp
        except RuntimeError:
            raise                          # propagate rate-limit error immediately
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    Retry {attempt}/{retries - 1} after {wait}s — {exc}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ── Step 1: Build CBSA → BLS series_id mapping from la.series catalog ────────
print("Step 1 — Building CBSA → series_id mapping from BLS la.series catalog ...")

# Load catalog from local cache if available; otherwise download and cache it.
# The cache avoids repeated downloads on re-runs.
if CATALOG_CACHE.exists():
    print(f"  Using cached catalog: {CATALOG_CACHE.name}")
    catalog_text = CATALOG_CACHE.read_text(encoding="utf-8", errors="replace")
else:
    print("  Downloading BLS la.series catalog (flat file, no API key required) ...")
    catalog_text = _fetch_catalog()
    CATALOG_CACHE.write_text(catalog_text, encoding="utf-8")
    print(f"  Cached to: {CATALOG_CACHE}")

# Parse: keep unadjusted (U) unemployment rate (03) for metro (B) and micro (D)
# Extract CBSA_CODE directly from area_code[4:9] — authoritative, no manual construction.
cbsa_to_series: dict[str, str] = {}
duplicate_cbsas: list[str] = []

for line in catalog_text.splitlines()[1:]:
    if not line.strip():
        continue
    parts = line.split("\t")
    if len(parts) < 5:
        continue
    series_id = parts[0].strip()
    area_type = parts[1].strip()
    area_code = parts[2].strip()
    measure   = parts[3].strip()
    seasonal  = parts[4].strip()

    if measure != "03" or seasonal != "U":
        continue
    if area_type not in ("B", "D"):
        continue
    if len(area_code) < 9:
        continue

    cbsa_code = area_code[4:9]

    if cbsa_code in cbsa_to_series:
        duplicate_cbsas.append(
            f"  WARNING: duplicate series for CBSA {cbsa_code}: "
            f"keeping {cbsa_to_series[cbsa_code]}, skipping {series_id}"
        )
        continue

    cbsa_to_series[cbsa_code] = series_id

for w in duplicate_cbsas:
    print(w)

print(f"  Series parsed from catalog : {len(cbsa_to_series)}")

# ── Load modeling table and check coverage ─────────────────────────────────────
v2_mac = pd.read_csv(V2_MAC_TABLE, dtype={"CBSA_CODE": str})
modeling_cbsas = sorted(v2_mac["CBSA_CODE"].unique())
n_modeling     = len(modeling_cbsas)
cbsa_title_map = (
    v2_mac[["CBSA_CODE", "CBSA_TITLE"]].drop_duplicates()
    .set_index("CBSA_CODE")["CBSA_TITLE"].to_dict()
)

covered = [c for c in modeling_cbsas if c in cbsa_to_series]
missing = [c for c in modeling_cbsas if c not in cbsa_to_series]
print(f"  Coverage : {len(covered)}/{n_modeling} modeling CBSAs")
if missing:
    for cbsa in missing:
        print(f"  *** MISSING from catalog: CBSA {cbsa}  {cbsa_title_map.get(cbsa, '?')} ***")
assert len(missing) == 0, f"Catalog missing series for modeling CBSAs: {missing}"
print("  [OK] All 855 modeling CBSAs found in catalog")

# ── Pre-run diagnostic for previously-failing CBSAs ───────────────────────────
DIAGNOSTIC_CBSAS = ["19340", "48260"]
print("\n  Series IDs for diagnostic CBSAs (were wrong in previous run):")
for cbsa in DIAGNOSTIC_CBSAS:
    sid   = cbsa_to_series.get(cbsa, "NOT FOUND")
    title = cbsa_title_map.get(cbsa, "?")
    print(f"    CBSA {cbsa}  {title:<45}  →  {sid}")

# ── Spot-check 5 known-good series via API ────────────────────────────────────
spot_cbsas = ["35620", "10420", "16980", "10220", "14460"]   # NYC, Akron, Chicago, Ada OK, Boston
spot_ids   = [cbsa_to_series[c] for c in spot_cbsas]
print(f"\n  Spot-checking 5 series against BLS API ...")
spot_resp = _post_bls(spot_ids, "2023", "2023")
spot_hits = {s["seriesID"]: len(s["data"]) for s in spot_resp.get("Results", {}).get("series", [])}
spot_ok = all(spot_hits.get(sid, 0) > 0 for sid in spot_ids)
for cbsa, sid in zip(spot_cbsas, spot_ids):
    n = spot_hits.get(sid, 0)
    mark = "✓" if n > 0 else "✗"
    print(f"    {mark} CBSA {cbsa}  {sid}  n={n}")
assert spot_ok, "Spot-check failed — verify series ID construction"
print("  [OK] All 5 spot-check series returned data")
time.sleep(1)


# ── Step 2: Fetch monthly data via BLS API v2 ─────────────────────────────────
print(f"\nStep 2 — Fetching monthly unemployment from BLS API v2 ...")
print(f"  Windows: {YEAR_WINDOWS}  |  Batch size: {BATCH_SIZE}")

# Build reverse lookup: series_id → CBSA_CODE (used when parsing API response)
series_to_cbsa: dict[str, str] = {v: k for k, v in cbsa_to_series.items()}

all_rows: list[dict] = []
api_requests_made = 0

series_list = [(cbsa, cbsa_to_series[cbsa]) for cbsa in modeling_cbsas]

for window_start, window_end in YEAR_WINDOWS:
    print(f"\n  Window {window_start}–{window_end}:")
    batches = [series_list[i:i + BATCH_SIZE]
               for i in range(0, len(series_list), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        series_ids = [b[1] for b in batch]

        resp = _post_bls(series_ids, window_start, window_end)
        api_requests_made += 1

        if resp.get("status") != "REQUEST_SUCCEEDED":
            msgs = resp.get("message", [])
            print(f"    Batch {batch_idx + 1}: WARNING status={resp.get('status')} msgs={msgs}")

        for series_result in resp.get("Results", {}).get("series", []):
            sid  = series_result["seriesID"]
            cbsa = series_to_cbsa.get(sid)
            if cbsa is None:
                print(f"    WARNING: unmapped series {sid}")
                continue

            for obs in series_result.get("data", []):
                period = obs.get("period", "")
                if not period.startswith("M") or period == "M13":
                    continue                        # skip annual summary M13
                value = obs.get("value", ".")
                if value == ".":
                    continue                        # BLS placeholder for missing
                try:
                    month = int(period[1:])         # "M01" → 1
                    rate  = float(value)
                except ValueError:
                    continue
                all_rows.append({
                    "CBSA_CODE":                 cbsa,
                    "YEAR":                      int(obs["year"]),
                    "MONTH":                     month,
                    "unemployment_rate_monthly": rate,
                })

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(batches):
            print(f"    Batch {batch_idx + 1}/{len(batches)} done — "
                  f"{api_requests_made} total requests, {len(all_rows):,} rows so far")
        time.sleep(1)

monthly_df = pd.DataFrame(all_rows)
monthly_df = monthly_df.sort_values(["CBSA_CODE", "YEAR", "MONTH"]).reset_index(drop=True)

cbsas_with_data = monthly_df["CBSA_CODE"].nunique()
if cbsas_with_data < n_modeling:
    missing_data = sorted(set(modeling_cbsas) - set(monthly_df["CBSA_CODE"].unique()))
    for cbsa in missing_data:
        sid   = cbsa_to_series.get(cbsa, "NOT IN CATALOG")
        title = cbsa_title_map.get(cbsa, "?")
        print(f"  *** MISSING DATA: CBSA {cbsa}  {title}  series={sid} ***")
    raise RuntimeError(
        f"Incomplete fetch: {cbsas_with_data}/{n_modeling} CBSAs have data.\n"
        "  Check series IDs for the missing CBSAs above and verify the catalog cache.\n"
        "  If the series ID is correct, delete the catalog cache and re-run to refresh it."
    )
print(f"  CBSAs with data: {cbsas_with_data}/{n_modeling}  [OK]")

monthly_df.to_csv(MONTHLY_OUT, index=False)
print(f"\n  Saved: {MONTHLY_OUT}  shape={monthly_df.shape}")


# ── Step 3: Aggregate to annual ───────────────────────────────────────────────
print(f"\nStep 3 — Aggregating to annual ...")

annual = (
    monthly_df
    .groupby(["CBSA_CODE", "YEAR"])["unemployment_rate_monthly"]
    .mean()
    .round(4)
    .rename("unemployment_rate")
    .reset_index()
)

annual = annual.sort_values(["CBSA_CODE", "YEAR"])
annual["unemployment_yoy"] = (
    annual.groupby("CBSA_CODE")["unemployment_rate"]
    .diff()
    .round(4)
)

# Download FRED UNRATE (national monthly → annual mean)
print("  Downloading FRED UNRATE ...")
unrate_raw = pd.read_csv(UNRATE_URL, parse_dates=["observation_date"])
unrate_raw["YEAR"] = unrate_raw["observation_date"].dt.year
national_annual = (
    unrate_raw.groupby("YEAR")["UNRATE"]
    .mean()
    .round(4)
    .rename("national_unrate_avg")
    .reset_index()
)
print(f"  UNRATE: {len(national_annual)} annual rows "
      f"({int(national_annual['YEAR'].min())}–{int(national_annual['YEAR'].max())})")

annual = annual.merge(national_annual, on="YEAR", how="left")
annual["unemployment_vs_national"] = (
    (annual["unemployment_rate"] - annual["national_unrate_avg"])
    .round(4)
)
annual = annual.drop(columns=["national_unrate_avg"])

annual.to_csv(ANNUAL_OUT, index=False)
print(f"  Saved: {ANNUAL_OUT}  shape={annual.shape}")


# ── Step 4: Merge into v2 macro modeling table ────────────────────────────────
print(f"\nStep 4 — Merging into Annual_Modeling_Table_v2_macro.csv ...")

collision = [c for c in NEW_COLS if c in v2_mac.columns]
assert len(collision) == 0, f"Column collision: {collision}"

v2_laus = v2_mac.merge(
    annual[["CBSA_CODE", "YEAR"] + NEW_COLS],
    on=["CBSA_CODE", "YEAR"],
    how="left",
)

assert len(v2_laus) == len(v2_mac), (
    f"Row count changed: v2_mac={len(v2_mac)}, v2_laus={len(v2_laus)}"
)

v2_laus.to_csv(LAUS_TABLE, index=False)
print(f"  Saved: {LAUS_TABLE}  shape={v2_laus.shape}")


# ── Step 5: Validation report ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("[1] Output files created")
for path in (MONTHLY_OUT, ANNUAL_OUT, LAUS_TABLE):
    size_mb = os.path.getsize(path) / 1e6
    cols    = len(pd.read_csv(path, dtype={"CBSA_CODE": str}, nrows=0).columns)
    rows    = sum(1 for _ in open(path)) - 1
    print(f"    {path.name:<50}  {rows:>8,} rows  {cols} cols  {size_mb:.1f} MB")

print(f"\n{SEP}")
print("[2] API coverage")
print(f"    Series derived (la.series catalog): {len(cbsa_to_series)}")
print(f"    Modeling CBSAs                    : {n_modeling}")
print(f"    Covered                           : {len(covered)}")
print(f"    Missing                           : {len(missing)}")
print(f"    Total API requests made           : {api_requests_made}")
print(f"    CBSAs with monthly data           : {cbsas_with_data}")

print(f"\n{SEP}")
print("[3] Year coverage")
for col in NEW_COLS:
    sub   = v2_laus.dropna(subset=[col])
    y_min = int(sub["YEAR"].min()) if len(sub) else "—"
    y_max = int(sub["YEAR"].max()) if len(sub) else "—"
    n_null = int(v2_laus[col].isna().sum())
    print(f"    {col:<35}  years {y_min}–{y_max}  nulls={n_null}")

print(f"\n{SEP}")
print("[4] Null check after merge")
for split_label, split_years in [
    ("train",   list(range(2010, 2022))),
    ("test",    list(range(2022, 2025))),
    ("predict", [2025]),
]:
    subset = v2_laus[v2_laus["YEAR"].isin(split_years)]
    print(f"    {split_label}:")
    for col in NEW_COLS:
        n  = int(subset[col].isna().sum())
        ok = "✓" if n == 0 else "✗ WARNING"
        print(f"      {col:<35}  {n} nulls  {ok}")

# unemployment_yoy null diagnosis
yoy_nulls      = v2_laus[v2_laus["unemployment_yoy"].isna()]
null_years     = sorted(yoy_nulls["YEAR"].unique().tolist())
first_year_per = v2_laus.groupby("CBSA_CODE")["YEAR"].min()
first_year_idx = set(
    zip(first_year_per.index, first_year_per.values)
)
yoy_null_idx = set(zip(yoy_nulls["CBSA_CODE"], yoy_nulls["YEAR"]))
only_first_year_nulls = yoy_null_idx.issubset(first_year_idx)
print(f"    unemployment_yoy null diagnosis:")
print(f"      Total nulls                    : {len(yoy_nulls)}")
print(f"      YEAR values with nulls         : {null_years}")
print(f"      Only first-year-per-CBSA nulls : {only_first_year_nulls}")

print(f"\n{SEP}")
print("[5] COVID unemployment spike check 2019–2024")
spike_years    = list(range(2019, 2025))
annual_check   = annual[annual["YEAR"].isin(spike_years)].copy()
national_check = national_annual[national_annual["YEAR"].isin(spike_years)].copy()
spike_summary  = (
    annual_check.groupby("YEAR")["unemployment_rate"]
    .agg(cbsa_mean="mean", cbsa_min="min", cbsa_max="max")
    .round(4)
    .reset_index()
    .merge(national_check.rename(columns={"national_unrate_avg": "national_unrate"}), on="YEAR")
)
print(spike_summary.to_string(index=False))
mean_2020 = float(spike_summary.loc[spike_summary["YEAR"] == 2020, "cbsa_mean"].iloc[0])
mean_2019 = float(spike_summary.loc[spike_summary["YEAR"] == 2019, "cbsa_mean"].iloc[0])
if mean_2020 > mean_2019 + 1.0:
    print(f"    ✓ 2020 cbsa_mean={mean_2020:.4f} clearly above 2019 cbsa_mean={mean_2019:.4f} — COVID spike confirmed")
else:
    print(f"    ✗ WARNING: 2020 spike not detected (2020={mean_2020:.4f}, 2019={mean_2019:.4f})")

print(f"\n{SEP}")
print("[6] Row count check")
assert len(v2_laus) == len(v2_mac), f"FAILED: v2_mac={len(v2_mac)}, v2_laus={len(v2_laus)}"
print(f"    Row count check PASSED: {len(v2_laus):,} rows")

print(f"\n{SEP}")
print("[7] Split integrity")
splits = {
    "train":   list(range(2010, 2022)),
    "test":    list(range(2022, 2025)),
    "predict": [2025],
}
for label, years in splits.items():
    n_mac  = int(v2_mac[v2_mac["YEAR"].isin(years)].shape[0])
    n_laus = int(v2_laus[v2_laus["YEAR"].isin(years)].shape[0])
    assert n_mac == n_laus, f"FAILED {label}: mac={n_mac}, laus={n_laus}"
    print(f"    {label:<10}  mac={n_mac:>6}  laus={n_laus:>6}  ✓")

print(f"\n{SEP}")
print("[8] Column check")
added = [c for c in v2_laus.columns if c not in v2_mac.columns]
print(f"    New columns added : {added}")
print(f"    v2_mac shape      : {v2_mac.shape}")
print(f"    v2_laus shape     : {v2_laus.shape}")

print(f"\n{SEP}")
print("Phase V2.3 complete.")
