"""
5_ModelingTable_Aggregation.py

Join the metro-level weather table (Stage 4 output) to all 9 Zillow ZHVI
variants and write a single modeling-ready CSV keyed on
(CBSA_CODE, YEAR_MONTH).

Zillow regions are mapped to Census CBSA codes via a name-shorten transform
('Atlanta-Sandy Springs-Roswell, GA' -> 'Atlanta, GA'), Unicode normalize,
and a small manual override file for the handful of cases the transform
cannot reach (Louisville, The Villages, etc.).

Inputs:
  - Methoden Data/Weather Data/Weather_Features_Metro.csv
  - 9 Zillow CSVs in Methoden Data/X_Original Data/Housing Data Original/
  - Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv
  - Methoden Data/X_Original Data/Other Data Original/Zillow_RegionID_to_CBSA_overrides.csv
  - Methoden Data/X_Original Data/Economic Data Original/bls_laus_cbsa_monthly_raw.csv
Output:
  - Methoden Data/Modeling_Table.csv  (~270k rows, 44 cols)
"""

from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEATHER_METRO_IN = ROOT / "Methoden Data/Weather Data/Weather_Features_Metro.csv"
HOUSING_DIR      = ROOT / "Methoden Data/X_Original Data/Housing Data Original"
DELIN_IN         = ROOT / "Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv"
OVERRIDES_IN     = ROOT / "Methoden Data/X_Original Data/Other Data Original/Zillow_RegionID_to_CBSA_overrides.csv"
BLS_LAUS_IN      = ROOT / "Methoden Data/X_Original Data/Economic Data Original/bls_laus_cbsa_monthly_raw.csv"
OUT_FILE         = ROOT / "Methoden Data/Modeling_Table.csv"

# Map each Zillow file to its destination column in the modeling table.
ZILLOW_FILES = {
    "Metro_zhvi_uc_all_homes_sfrcondo_botom_tier_0.0_0.33_sm_sa_month.csv": "zhvi_all_bottom",
    "Metro_zhvi_uc_all_homes_sfrcondo_toptier_0.67_1.0_sm_sa_month.csv":   "zhvi_all_top",
    "Metro_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv":                    "zhvi_sfr_mid",
    "Metro_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv":                  "zhvi_condo_mid",
    "Metro_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":     "zhvi_1br_mid",
    "Metro_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":     "zhvi_2br_mid",
    "Metro_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":     "zhvi_3br_mid",
    "Metro_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":     "zhvi_4br_mid",
    "Metro_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":     "zhvi_5br_mid",
}
ZHVI_COLS = list(ZILLOW_FILES.values())


def normalize_text(s):
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def shorten_cbsa_title(title):
    if not isinstance(title, str):
        return title
    m = re.match(r"^([^-,]+)(?:-[^,]+)?,\s*([A-Z]{2})(?:-[A-Z]{2})*$", title)
    return f"{m.group(1)}, {m.group(2)}" if m else title


def build_zillow_to_cbsa_map():
    delin = pd.read_csv(DELIN_IN, skiprows=2, dtype=str)
    keep = delin["Metropolitan/Micropolitan Statistical Area"].isin([
        "Metropolitan Statistical Area",
        "Micropolitan Statistical Area",
    ])
    pairs = delin.loc[keep, ["CBSA Code", "CBSA Title"]].dropna().drop_duplicates()
    pairs["short"] = pairs["CBSA Title"].apply(shorten_cbsa_title).apply(normalize_text)
    short_to_code = dict(zip(pairs["short"], pairs["CBSA Code"]))

    overrides = pd.read_csv(OVERRIDES_IN, dtype=str)
    overrides["RegionID"] = overrides["RegionID"].astype(str)
    rid_to_code = dict(zip(overrides["RegionID"], overrides["CBSA_CODE"]))

    return short_to_code, rid_to_code


def map_zillow_region(region_id, region_name, short_to_code, rid_to_code):
    rid = str(region_id)
    if rid in rid_to_code:
        return rid_to_code[rid]
    norm = normalize_text(region_name) if region_name else None
    return short_to_code.get(norm)


def read_bls_laus(path):
    """Load BLS LAUS monthly unemployment rate keyed on (CBSA_CODE, YEAR_MONTH)."""
    df = pd.read_csv(path, dtype={"CBSA_CODE": str})
    df["YEAR_MONTH"] = (
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2)
    )
    return df[["CBSA_CODE", "YEAR_MONTH", "unemployment_rate_monthly"]]


def read_zillow_long(path, value_col):
    """Read a wide-format Zillow file and melt to long: (RegionID, RegionName, YEAR_MONTH, value_col)."""
    df = pd.read_csv(path)
    id_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    id_cols = [c for c in id_cols if c in df.columns]
    df = df[df["RegionType"] == "msa"]
    month_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}$", str(c))]
    long = df.melt(id_vars=id_cols, value_vars=month_cols,
                   var_name="month_end", value_name=value_col)
    long = long.dropna(subset=[value_col]).copy()
    long["YEAR_MONTH"] = long["month_end"].str[:7]
    long["RegionID"] = long["RegionID"].astype(str)
    return long[["RegionID", "RegionName", "YEAR_MONTH", value_col]]


def main():
    print(f"[Stage 5] reading {WEATHER_METRO_IN.name} ...")
    weather = pd.read_csv(WEATHER_METRO_IN, dtype={"CBSA_CODE": str})
    print(f"  weather: {len(weather):,} rows, {weather['CBSA_CODE'].nunique():,} CBSAs, "
          f"months {weather['YEAR_MONTH'].min()}..{weather['YEAR_MONTH'].max()}")

    short_to_code, rid_to_code = build_zillow_to_cbsa_map()
    print(f"  CBSA crosswalk: {len(short_to_code):,} shortened-title entries, "
          f"{len(rid_to_code):,} manual RegionID overrides")

    zillow_combined = None

    for filename, col_name in ZILLOW_FILES.items():
        path = HOUSING_DIR / filename
        print(f"\n  [{col_name}] reading {filename} ...")
        long = read_zillow_long(path, col_name)
        rows = len(long)

        # Map RegionID -> CBSA_CODE once per unique region, then broadcast.
        regions = long[["RegionID", "RegionName"]].drop_duplicates()
        regions["CBSA_CODE"] = regions.apply(
            lambda r: map_zillow_region(r["RegionID"], r["RegionName"], short_to_code, rid_to_code),
            axis=1,
        )
        unmapped_regions = regions.loc[regions["CBSA_CODE"].isna(), "RegionName"].tolist()
        rid_to_cbsa = dict(zip(regions["RegionID"], regions["CBSA_CODE"]))
        long["CBSA_CODE"] = long["RegionID"].map(rid_to_cbsa)
        long = long.dropna(subset=["CBSA_CODE"])[["CBSA_CODE", "YEAR_MONTH", col_name]].copy()

        # If multiple RegionIDs map to the same CBSA (shouldn't happen with our overrides
        # but defensive), take the first non-null per cell.
        long = long.groupby(["CBSA_CODE", "YEAR_MONTH"], as_index=False)[col_name].first()

        print(f"    {rows:,} long rows -> {long['CBSA_CODE'].nunique():,} CBSAs mapped, "
              f"{len(unmapped_regions):,} regions unmapped")
        if unmapped_regions and zillow_combined is None:
            print(f"    sample unmapped (max 5): {unmapped_regions[:5]}")

        if zillow_combined is None:
            zillow_combined = long
        else:
            zillow_combined = zillow_combined.merge(
                long, on=["CBSA_CODE", "YEAR_MONTH"], how="outer"
            )

    print(f"\n  combined Zillow frame: {len(zillow_combined):,} rows, "
          f"{zillow_combined['CBSA_CODE'].nunique():,} CBSAs")

    # BLS LAUS monthly unemployment (CBSA-level, 2009+) — left-joined onto
    # weather first so the column lands between precip_in and the zhvi_* cols.
    # Leaves NaN for 2000-2008 since the LAUS metro series doesn't reach back.
    print(f"\n  reading {BLS_LAUS_IN.name} ...")
    bls = read_bls_laus(BLS_LAUS_IN)
    print(f"    {len(bls):,} rows, {bls['CBSA_CODE'].nunique():,} CBSAs, "
          f"months {bls['YEAR_MONTH'].min()}..{bls['YEAR_MONTH'].max()}")
    weather = weather.merge(bls, on=["CBSA_CODE", "YEAR_MONTH"], how="left")

    # Inner-join the weather metro table to the Zillow frame on (CBSA_CODE, YEAR_MONTH).
    out = weather.merge(zillow_combined, on=["CBSA_CODE", "YEAR_MONTH"], how="inner")
    out = out[out["YEAR_MONTH"] >= "2000-01"].copy()
    out = out.sort_values(["CBSA_CODE", "YEAR_MONTH"]).reset_index(drop=True)

    # Forward 6-month lookups — prediction targets. Per-CBSA date-based lookup
    # (not .shift(-6) positional) of every ZHVI column 6 months later, plus the
    # row-wise mean across the 9 categories for an aggregate % change. Date-based
    # so any month gaps don't produce a wrong "6 months ahead" anchor.
    out["_zhvi_avg"] = out[ZHVI_COLS].mean(axis=1)
    out["_dt"] = pd.to_datetime(out["YEAR_MONTH"] + "-01")
    future_cols = ZHVI_COLS + ["_zhvi_avg"]
    future = out[["CBSA_CODE", "_dt"] + future_cols].rename(
        columns={"_dt": "_dt_then", **{c: f"__future__{c}" for c in future_cols}}
    )
    out["_dt_lookup"] = out["_dt"] + pd.DateOffset(months=6)
    out = out.merge(
        future,
        left_on=["CBSA_CODE", "_dt_lookup"],
        right_on=["CBSA_CODE", "_dt_then"],
        how="left",
    )
    for c in ZHVI_COLS:
        out[f"{c}_next_6m"] = out[f"__future__{c}"]
    out["zhvi_avg_next_6m"] = out["__future___zhvi_avg"]
    # Fraction, not percent: 1.0 == +100%, 0.01 == +1%, -0.05 == -5%.
    out["price_change_next_6m"] = (
        (out["zhvi_avg_next_6m"] - out["_zhvi_avg"]) / out["_zhvi_avg"]
    )
    out = out.drop(columns=(
        ["_zhvi_avg", "_dt", "_dt_lookup", "_dt_then", "__future___zhvi_avg"]
        + [f"__future__{c}" for c in ZHVI_COLS]
    ))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"\n  wrote {OUT_FILE.name} -- {len(out):,} rows, {len(out.columns)} cols, {size_mb:.1f} MB")
    print(f"  CBSAs in output: {out['CBSA_CODE'].nunique():,}")

    print("\n[Non-null counts per ZHVI column]")
    for c in ZHVI_COLS:
        n = out[c].notna().sum()
        print(f"  {c:18s}  {n:>9,}  ({100*n/len(out):5.1f}%)")

    n_unemp = out["unemployment_rate_monthly"].notna().sum()
    print(f"\n[BLS LAUS unemployment_rate_monthly] "
          f"{n_unemp:,} non-null ({100*n_unemp/len(out):.1f}%) — NaN expected for 2000-2008")

    n_pct = out["price_change_next_6m"].notna().sum()
    print(f"\n[price_change_next_6m] "
          f"{n_pct:,} non-null ({100*n_pct/len(out):.1f}%) — NaN for the last 6 months per CBSA")

    print("\n[Non-null counts per *_next_6m column]")
    for c in ZHVI_COLS:
        n = out[f"{c}_next_6m"].notna().sum()
        print(f"  {c+'_next_6m':28s}  {n:>9,}  ({100*n/len(out):5.1f}%)")

    # Worked example
    print("\n[Worked example] Atlanta CBSA 12060, 2024-09:")
    atl = out[(out["CBSA_CODE"] == "12060") & (out["YEAR_MONTH"] == "2024-09")]
    if len(atl):
        for col, val in atl.iloc[0].items():
            print(f"  {col:30s}  {val}")
    else:
        print("  (no row found)")

    print(f"\n[Stage 5] done.")


if __name__ == "__main__":
    main()
