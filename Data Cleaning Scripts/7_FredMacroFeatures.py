"""
Phase V2.1 — FRED Macro Feature Pipeline
=========================================

PURPOSE
-------
Download national macroeconomic time series from FRED, aggregate them to
annual frequency, and left-join them into the annual modeling table to create
a v2-ready input file.  This is a data-preparation step only; no model is
trained here.  Economic variables are added nationally (broadcast to all CBSAs)
as Phase V2.1; CBSA-level variables (e.g. BLS unemployment) are Phase V2.2.

DATA SOURCES
------------
FRED MORTGAGE30US
  URL      : https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US
  Grain    : Weekly (Thursday observations)
  Coverage : 1971-04-02 to present
  Aggregation : Annual mean of weekly observations (YEAR <= 2025)
  Join key : YEAR (national series, broadcast to all CBSAs)

FRED CPIAUCSL
  URL      : https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL
  Grain    : Monthly
  Coverage : 1947-01-01 to present
  Aggregation : Annual mean of monthly observations (YEAR <= 2025)
  Join key : YEAR (national series, broadcast to all CBSAs)

FEATURES PRODUCED
-----------------
mortgage_rate_avg  annual average 30-yr fixed mortgage rate (percentage)
mortgage_rate_yoy  YoY change in mortgage_rate_avg (percentage points)
cpi_avg            annual average CPI index level [audit only — non-stationary]
cpi_yoy_pct        YoY % change in cpi_avg (annual headline inflation rate)

LEAKAGE LOGIC
-------------
Macro features for YEAR=t describe the macroeconomic environment during year t.
The modeling target target_growth_1y measures Dec(t) to Dec(t+1) ZHVI growth.
Therefore macro_t predicts growth_{t→t+1} and does not use any future t+1
information.  Do not shift macro variables forward.

INPUTS  (read-only)
-------------------
  Methoden Data/Modeling Data/Annual_Modeling_Table.csv

OUTPUTS  (new files only — no existing file is modified)
---------------------------------------------------------
  Methoden Data/Economic Data/fred_mortgage_rate_annual.csv
  Methoden Data/Economic Data/fred_cpi_annual.csv
  Methoden Data/Modeling Data/Annual_Modeling_Table_v2_macro.csv
"""

from pathlib import Path
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
ECON_DIR = ROOT / "Methoden Data" / "Economic Data"
ECON_DIR.mkdir(parents=True, exist_ok=True)

V1_TABLE = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table.csv"
V2_TABLE = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table_v2_macro.csv"
MORT_OUT = ECON_DIR / "fred_mortgage_rate_annual.csv"
CPI_OUT  = ECON_DIR / "fred_cpi_annual.csv"

MORT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
CPI_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"

YEAR_MAX = 2025

# ── Step 1: MORTGAGE30US ──────────────────────────────────────────────────────
print("Downloading MORTGAGE30US ...")
mort_raw = pd.read_csv(MORT_URL, parse_dates=["observation_date"])
mort_raw["YEAR"] = mort_raw["observation_date"].dt.year
mort_raw = mort_raw[mort_raw["YEAR"] <= YEAR_MAX].copy()

obs_per_year = mort_raw.groupby("YEAR")["MORTGAGE30US"].count()
thin = obs_per_year[obs_per_year < 45]
if len(thin) > 0:
    print(f"  WARNING: years with < 45 weekly observations: {thin.to_dict()}")

mort_annual = (
    mort_raw.groupby("YEAR")["MORTGAGE30US"]
    .mean()
    .round(4)
    .rename("mortgage_rate_avg")
    .reset_index()
)
mort_annual["mortgage_rate_yoy"] = mort_annual["mortgage_rate_avg"].diff().round(4)
mort_annual.to_csv(MORT_OUT, index=False)
print(f"  Saved: {MORT_OUT}  ({len(mort_annual)} rows)")

# ── Step 2: CPIAUCSL ──────────────────────────────────────────────────────────
print("Downloading CPIAUCSL ...")
cpi_raw = pd.read_csv(CPI_URL, parse_dates=["observation_date"])
cpi_raw["YEAR"] = cpi_raw["observation_date"].dt.year
cpi_raw = cpi_raw[cpi_raw["YEAR"] <= YEAR_MAX].copy()

obs_per_year_cpi = cpi_raw.groupby("YEAR")["CPIAUCSL"].count()
thin_cpi = obs_per_year_cpi[obs_per_year_cpi < 12]
if len(thin_cpi) > 0:
    print(f"  WARNING: years with < 12 monthly CPI observations: {thin_cpi.to_dict()}")

cpi_annual = (
    cpi_raw.groupby("YEAR")["CPIAUCSL"]
    .mean()
    .round(3)
    .rename("cpi_avg")
    .reset_index()
)
# cpi_avg is kept for audit purposes only — it is a non-stationary ever-rising
# index level and must NOT be used as a direct model feature.
# cpi_yoy_pct (percentage change) is the stationary, model-ready feature.
cpi_annual["cpi_yoy_pct"] = (cpi_annual["cpi_avg"].pct_change() * 100).round(4)
cpi_annual.to_csv(CPI_OUT, index=False)
print(f"  Saved: {CPI_OUT}  ({len(cpi_annual)} rows)")

# ── Step 3: Merge ──────────────────────────────────────────────────────────────
print("\nLoading Annual_Modeling_Table.csv (read-only) ...")
v1 = pd.read_csv(V1_TABLE, dtype={"CBSA_CODE": str})

# Guard: none of the four new columns may already exist in the v1 table
new_cols = ["mortgage_rate_avg", "mortgage_rate_yoy", "cpi_avg", "cpi_yoy_pct"]
collision = [c for c in new_cols if c in v1.columns]
assert len(collision) == 0, f"Column name collision detected in v1 table: {collision}"

# Inner-join the two macro intermediates on YEAR (identical YEAR coverage expected)
macro_df = mort_annual.merge(cpi_annual, on="YEAR", how="inner")

# Left-join macro into the modeling table on YEAR.
# Leakage guard: macro features for YEAR=t describe conditions fully known by
# Dec 31 of year t.  target_growth_1y covers Dec t → Dec t+1.
# Do not shift macro variables forward.
v2 = v1.merge(macro_df, on="YEAR", how="left")

v2.to_csv(V2_TABLE, index=False)
print(f"  Saved: {V2_TABLE}  shape={v2.shape}")

# ── Step 4: Validation report ─────────────────────────────────────────────────
SEP = "─" * 62

print(f"\n{SEP}")
print("[1] Output files created")
print(f"    {MORT_OUT}")
print(f"    {CPI_OUT}")
print(f"    {V2_TABLE}")

print(f"\n{SEP}")
print("[2] Year coverage")
print(f"    mortgage_rate_avg : {int(mort_annual['YEAR'].min())} – {int(mort_annual['YEAR'].max())}"
      f"  |  nulls: {mort_annual['mortgage_rate_avg'].isna().sum()}")
print(f"    mortgage_rate_yoy : {int(mort_annual['YEAR'].min())} – {int(mort_annual['YEAR'].max())}"
      f"  |  nulls: {mort_annual['mortgage_rate_yoy'].isna().sum()}"
      f"  (1 expected for earliest year)")
print(f"    cpi_avg           : {int(cpi_annual['YEAR'].min())} – {int(cpi_annual['YEAR'].max())}"
      f"  |  nulls: {cpi_annual['cpi_avg'].isna().sum()}")
print(f"    cpi_yoy_pct       : {int(cpi_annual['YEAR'].min())} – {int(cpi_annual['YEAR'].max())}"
      f"  |  nulls: {cpi_annual['cpi_yoy_pct'].isna().sum()}"
      f"  (1 expected for earliest year)")

print(f"\n{SEP}")
print("[3] Missing values after merge (v2 table)")
for col in new_cols:
    n_null = v2[col].isna().sum()
    status = "✓" if n_null == 0 else "✗ WARNING"
    print(f"    {col:<25} {n_null} nulls  {status}")

print(f"\n{SEP}")
print("[4] 2020–2024 macro spike check")
spike = (
    v2[v2["YEAR"].isin([2020, 2021, 2022, 2023, 2024])]
    [["YEAR", "mortgage_rate_avg", "mortgage_rate_yoy", "cpi_yoy_pct"]]
    .drop_duplicates("YEAR")
    .sort_values("YEAR")
    .reset_index(drop=True)
)
print(spike.to_string(index=False))
yoy_2022 = float(spike.loc[spike["YEAR"] == 2022, "mortgage_rate_yoy"].iloc[0])
if yoy_2022 > 1.0:
    print(f"    ✓ mortgage_rate_yoy 2022 = {yoy_2022:+.4f} pp  (strongly positive — rate spike confirmed)")
else:
    print(f"    ✗ WARNING: mortgage_rate_yoy 2022 = {yoy_2022:+.4f} pp  (expected > +1.0 pp)")

print(f"\n{SEP}")
print("[5] Row count check")
assert len(v2) == len(v1), f"FAILED: v1={len(v1)} rows, v2={len(v2)} rows"
print(f"    Row count check PASSED: {len(v2)} rows")

print(f"\n{SEP}")
print("[6] Split integrity")
splits = {
    "train":   list(range(2010, 2022)),
    "test":    list(range(2022, 2025)),
    "predict": [2025],
}
for label, years in splits.items():
    n_v1 = int(v1[v1["YEAR"].isin(years)].shape[0])
    n_v2 = int(v2[v2["YEAR"].isin(years)].shape[0])
    assert n_v1 == n_v2, f"FAILED {label}: v1={n_v1}, v2={n_v2}"
    print(f"    {label:<10} v1={n_v1:>6}  v2={n_v2:>6}  ✓")

print(f"\n{SEP}")
print("[7] Column check")
added = [c for c in v2.columns if c not in v1.columns]
print(f"    New columns added  : {added}")
print(f"    v1 shape           : {v1.shape}")
print(f"    v2 shape           : {v2.shape}")

print(f"\n{SEP}")
print("Phase V2.1 complete.")
