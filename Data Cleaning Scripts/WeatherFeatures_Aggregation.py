"""
Build the unified county × month weather-features table by aggregating
StormEvents (NOAA) and the cleaned FEMA disaster declarations.

Output (Methoden Data/Weather Data/Weather_Features.csv) joins directly to
Zillow on (STATE_FIPS, COUNTY_FIPS, YEAR_MONTH).

Design choices, documented inline:
  1. StormEvents rows with CZ_TYPE='Z' (NWS forecast zones, ~half the file)
     have a zone code, not a county FIPS, in CZ_FIPS — without a zone→county
     crosswalk we can't put their damage/death numbers in the right county.
     We DO use them for state-level hazard flags though: had_hurricane,
     had_drought, had_winter_storm etc. are OR-aggregated across all Z-zone
     events in the same (state, month) and then propagated to every county in
     that state-month. So the flags are representative even though damage and
     death counts remain county-level (C-rows only).
  2. FEMA rows tagged is_statewide=True (fipsCountyCode='000') are dropped —
     attributing them to a single county is wrong, and exploding them across
     all counties of the state without a population/area weighting would over-
     count. Few rows, low-stakes either way.
  3. FEMA declarations are exploded across the months their incident was
     active so multi-month events contribute to every month they spanned.
     `fema_active_days` records how many days of the month were under the
     declaration; binary flags use OR-aggregation.

Run order: this script depends on StormEvents_ALL.csv and
Fema_DisasterDeclarations_Cleaned.csv. Run StormEvents_Aggregation.py and
Fema_DisasterDeclarations_Cleaning.py first.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORM_SRC = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Storm" / "StormEvents_ALL.csv"
FEMA_SRC = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Fema" / "Fema_DisasterDeclarations_Cleaned.csv"
FEMA_ORIG = PROJECT_ROOT / "Methoden Data" / "X_Original Data" / "Weather Data Original" / "Fema_DisasterDeclarationsSummaries.csv"
TEMP_SRC = PROJECT_ROOT / "Methoden Data" / "X_Original Data" / "Weather Data Original" / "Temp_per_county_month.csv"
OUT = PROJECT_ROOT / "Methoden Data" / "Weather Data" / "Weather_Features.csv"

# Census STATE_FIPS codes for U.S. territories that NOAA Storm Events does not
# cover (NOAA uses its own non-standard codes 96–99 for these). FEMA uses the
# Census codes, so we add these names manually.
TERRITORY_STATE_NAMES = {
    60: "AMERICAN SAMOA",
    64: "FEDERATED STATES OF MICRONESIA",
    66: "GUAM",
    68: "MARSHALL ISLANDS",
    69: "NORTHERN MARIANA ISLANDS",
    70: "PALAU",
    72: "PUERTO RICO",
    74: "U.S. MINOR OUTLYING ISLANDS",
    78: "U.S. VIRGIN ISLANDS",
}

STORM_HAZARDS = {
    "had_tornado": {"Tornado", "Funnel Cloud"},
    "had_hurricane": {"Hurricane", "Hurricane (Typhoon)", "Tropical Storm", "Tropical Depression"},
    "had_flood": {"Flood", "Flash Flood", "Coastal Flood", "Lakeshore Flood"},
    "had_drought": {"Drought"},
    "had_heat": {"Heat", "Excessive Heat"},
    "had_winter_storm": {"Winter Storm", "Blizzard", "Ice Storm", "Heavy Snow"},
    "had_wildfire": {"Wildfire"},
}

FEMA_HAZARDS = {
    "had_fema_flood": {"Flood"},
    "had_fema_hurricane": {"Hurricane", "Tropical Storm", "Typhoon"},
    "had_fema_severe_storm": {"Severe Storm", "Severe Ice Storm", "Snowstorm"},
    "had_fema_fire": {"Fire"},
    "had_fema_tornado": {"Tornado"},
    "had_fema_earthquake": {"Earthquake"},
    "had_fema_biological": {"Biological"},
}

_TOR_SCALE_RE = re.compile(r"^E?F(\d)$")


def build_name_lookups(storm_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull (STATE_FIPS -> STATE) and (STATE_FIPS, COUNTY_FIPS) -> COUNTY name
    lookups out of the storm file. State names come from any row; county names
    only from CZ_TYPE='C' rows (Z-rows hold zone names, not county names).
    """
    df = pd.read_csv(
        storm_path,
        low_memory=False,
        usecols=["STATE", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "CZ_NAME"],
    )
    state_map = (
        df[["STATE_FIPS", "STATE"]]
        .dropna()
        .drop_duplicates(subset=["STATE_FIPS"])
        .reset_index(drop=True)
    )
    present = set(state_map["STATE_FIPS"])
    extra_rows = [
        {"STATE_FIPS": fips, "STATE": name}
        for fips, name in TERRITORY_STATE_NAMES.items()
        if fips not in present
    ]
    if extra_rows:
        state_map = pd.concat([state_map, pd.DataFrame(extra_rows)], ignore_index=True)

    county_map = (
        df[df["CZ_TYPE"] == "C"][["STATE_FIPS", "CZ_FIPS", "CZ_NAME"]]
        .dropna()
        .drop_duplicates(subset=["STATE_FIPS", "CZ_FIPS"])
        .rename(columns={"CZ_FIPS": "COUNTY_FIPS", "CZ_NAME": "COUNTY"})
        .reset_index(drop=True)
    )
    return state_map, county_map


def aggregate_temp(path: Path) -> pd.DataFrame:
    """Load monthly per-county temperature & precipitation. The file is
    already at the right grain — just rename keys and build YEAR_MONTH."""
    print(f"Loading {path.name}")
    df = pd.read_csv(
        path,
        low_memory=False,
        usecols=[
            "state_fips",
            "county_fips",
            "year",
            "month",
            "tmax_f",
            "tmin_f",
            "precip_in",
        ],
    )
    n_in = len(df)
    df["YEAR_MONTH"] = (
        df["year"].astype(str).str.zfill(4) + "-" + df["month"].astype(str).str.zfill(2)
    )
    df = df.rename(columns={"state_fips": "STATE_FIPS", "county_fips": "COUNTY_FIPS"})
    df = df[["STATE_FIPS", "COUNTY_FIPS", "YEAR_MONTH", "tmax_f", "tmin_f", "precip_in"]]
    n_counties = df.drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"]).shape[0]
    print(f"  {n_in:,} rows, {df['STATE_FIPS'].nunique()} states, {n_counties:,} counties")
    return df


def temp_county_name_fallback(temp_path: Path) -> pd.DataFrame:
    """Pull (STATE_FIPS, COUNTY_FIPS) -> COUNTY name from the temp file's
    `county_name` column. Strips the trailing local-government suffix
    (' County', ' Parish', ' Borough', ' Census Area', ' Municipality',
    ' City and Borough') and uppercases to match the storm/FEMA style.
    """
    df = pd.read_csv(
        temp_path,
        low_memory=False,
        usecols=["state_fips", "county_fips", "county_name"],
    )
    df = df.rename(columns={"state_fips": "STATE_FIPS", "county_fips": "COUNTY_FIPS"})
    df["COUNTY_TEMP"] = (
        df["county_name"]
        .str.replace(
            r"\s+(County|Parish|Borough|Census Area|Municipality|City and Borough)$",
            "",
            regex=True,
        )
        .str.strip()
        .str.upper()
    )
    return (
        df[["STATE_FIPS", "COUNTY_FIPS", "COUNTY_TEMP"]]
        .dropna()
        .drop_duplicates(subset=["STATE_FIPS", "COUNTY_FIPS"])
        .reset_index(drop=True)
    )


def fema_county_name_fallback(fema_orig_path: Path) -> pd.DataFrame:
    """Pull (STATE_FIPS, COUNTY_FIPS) -> COUNTY name from the original FEMA
    file's `designatedArea` field, used to fill counties that NOAA never
    reported on (mostly territories and a handful of Alaska boroughs).
    """
    df = pd.read_csv(
        fema_orig_path,
        low_memory=False,
        dtype=str,
        usecols=["fipsStateCode", "fipsCountyCode", "designatedArea"],
    )
    df = df[df["fipsCountyCode"] != "000"].copy()
    df["STATE_FIPS"] = df["fipsStateCode"].astype(int)
    df["COUNTY_FIPS"] = df["fipsCountyCode"].astype(int)
    df["COUNTY_FALLBACK"] = (
        df["designatedArea"]
        .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
        .str.strip()
        .str.upper()
    )
    return (
        df[["STATE_FIPS", "COUNTY_FIPS", "COUNTY_FALLBACK"]]
        .dropna()
        .drop_duplicates(subset=["STATE_FIPS", "COUNTY_FIPS"])
        .reset_index(drop=True)
    )


def parse_tor_scale(s: pd.Series) -> pd.Series:
    out = np.full(len(s), np.nan, dtype="float64")
    arr = s.astype("string").to_numpy()
    for i, raw in enumerate(arr):
        if raw is pd.NA or raw is None:
            continue
        m = _TOR_SCALE_RE.match(str(raw).strip())
        if m:
            out[i] = float(m.group(1))
    return pd.Series(out, index=s.index, dtype="float64")


def aggregate_storm(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (county_features, state_zone_flags).

    county_features: per (STATE_FIPS, COUNTY_FIPS, YEAR_MONTH), built from
    CZ_TYPE='C' rows only — damage/death counts and per-county hazard flags.

    state_zone_flags: per (STATE_FIPS, YEAR_MONTH), built from CZ_TYPE='Z'
    rows only — boolean hazard flags (one column per STORM_HAZARDS entry,
    prefixed with `_state_` so it's clear they are merged in temporarily).
    Damage/death counts from Z-rows are intentionally not propagated, since
    we cannot attribute them to a specific county without a zone crosswalk.
    """
    print(f"Loading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    n_in = len(df)

    df["YEAR_MONTH"] = df["BEGIN_DATE"].astype(str).str[:7]
    df["TOR_F_SCALE_NUM"] = parse_tor_scale(df["TOR_F_SCALE"])
    df["DEATHS_TOTAL"] = df["DEATHS_DIRECT"].fillna(0) + df["DEATHS_INDIRECT"].fillna(0)
    for col, types in STORM_HAZARDS.items():
        df[col] = df["EVENT_TYPE"].isin(types)

    c = df[df["CZ_TYPE"] == "C"]
    z = df[df["CZ_TYPE"] == "Z"]
    print(f"  {n_in:,} rows in")
    print(f"    {len(c):,} C-rows -> county-level damage/deaths/flags")
    print(f"    {len(z):,} Z-rows -> state-level hazard-flag fallback")

    county_agg = {
        "n_storm_events": ("EVENT_TYPE", "count"),
        "damage_property_sum": ("DAMAGE_PROPERTY", "sum"),
        "damage_crops_sum": ("DAMAGE_CROPS", "sum"),
        "deaths_total": ("DEATHS_TOTAL", "sum"),
        "injuries_total": ("INJURIES_DIRECT", "sum"),
        **{col: (col, "max") for col in STORM_HAZARDS},
    }
    county_features = c.groupby(["STATE_FIPS", "CZ_FIPS", "YEAR_MONTH"], as_index=False).agg(**county_agg)
    county_features = county_features.rename(columns={"CZ_FIPS": "COUNTY_FIPS"})

    state_zone_flags = z.groupby(["STATE_FIPS", "YEAR_MONTH"], as_index=False).agg(
        **{f"_state_{col}": (col, "max") for col in STORM_HAZARDS}
    )

    return county_features, state_zone_flags


def aggregate_fema(path: Path) -> pd.DataFrame:
    print(f"Loading {path.name}")
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={"fipsStateCode": str, "fipsCountyCode": str},
    )
    n_in = len(df)

    is_statewide = df["is_statewide"].astype(str).str.lower() == "true"
    df = df[~is_statewide].copy()
    print(f"  {n_in:,} rows in -> {len(df):,} county rows (statewide rows excluded)")

    df["incidentBeginDate"] = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
    df["incidentEndDate"] = pd.to_datetime(df["incidentEndDate"], errors="coerce")
    df["incidentEndDate"] = df["incidentEndDate"].fillna(df["incidentBeginDate"])

    rows = []
    for r in df.itertuples(index=False):
        if pd.isna(r.incidentBeginDate):
            continue
        months = pd.period_range(r.incidentBeginDate, r.incidentEndDate, freq="M")
        for m in months:
            month_start = m.to_timestamp()
            month_end = (m + 1).to_timestamp() - pd.Timedelta(days=1)
            active_start = max(r.incidentBeginDate, month_start)
            active_end = min(r.incidentEndDate, month_end)
            active_days = (active_end - active_start).days + 1
            rows.append(
                {
                    "STATE_FIPS": int(r.fipsStateCode),
                    "COUNTY_FIPS": int(r.fipsCountyCode),
                    "YEAR_MONTH": str(m),
                    "active_days": active_days,
                    "declarationType": r.declarationType,
                    "incidentType": r.incidentType,
                    "iaProgramDeclared": r.iaProgramDeclared,
                    "ihProgramDeclared": r.ihProgramDeclared,
                    "paProgramDeclared": r.paProgramDeclared,
                    "hmProgramDeclared": r.hmProgramDeclared,
                }
            )
    exp = pd.DataFrame(rows)
    print(f"  {len(df):,} declarations exploded to {len(exp):,} (county × month) rows")

    exp["is_DR"] = exp["declarationType"] == "DR"
    exp["is_EM"] = exp["declarationType"] == "EM"
    exp["is_FM"] = exp["declarationType"] == "FM"
    for col, types in FEMA_HAZARDS.items():
        exp[col] = exp["incidentType"].isin(types)

    agg = {
        "n_fema_declarations": ("declarationType", "count"),
        "fema_active_days": ("active_days", "sum"),
        "had_major_disaster": ("is_DR", "max"),
        "had_emergency": ("is_EM", "max"),
        "had_fire_mgmt": ("is_FM", "max"),
        "ia_active": ("iaProgramDeclared", "max"),
        "ih_active": ("ihProgramDeclared", "max"),
        "pa_active": ("paProgramDeclared", "max"),
        "hm_active": ("hmProgramDeclared", "max"),
        **{col: (col, "max") for col in FEMA_HAZARDS},
    }
    grouped = exp.groupby(["STATE_FIPS", "COUNTY_FIPS", "YEAR_MONTH"], as_index=False).agg(**agg)
    return grouped


def main() -> None:
    storm_c, storm_z = aggregate_storm(STORM_SRC)
    fema = aggregate_fema(FEMA_SRC)
    temp = aggregate_temp(TEMP_SRC)

    print(f"\nStorm county features: {len(storm_c):,} rows, {storm_c.shape[1]} cols")
    print(f"Storm zone state flags: {len(storm_z):,} (state, month) rows")
    print(f"FEMA features:          {len(fema):,} rows, {fema.shape[1]} cols")
    print(f"Temp features:          {len(temp):,} rows, {temp.shape[1]} cols")

    merged = storm_c.merge(fema, on=["STATE_FIPS", "COUNTY_FIPS", "YEAR_MONTH"], how="outer")
    merged = merged.merge(storm_z, on=["STATE_FIPS", "YEAR_MONTH"], how="left")
    merged = merged.merge(temp, on=["STATE_FIPS", "COUNTY_FIPS", "YEAR_MONTH"], how="outer")

    # Propagate state-level Z-zone hazard flags onto every county-month row
    # in that state-month, OR-combining with the existing C-row flag.
    for col in STORM_HAZARDS:
        state_col = f"_state_{col}"
        if state_col in merged.columns:
            merged[col] = (
                merged[col].fillna(False).astype(bool)
                | merged[state_col].fillna(False).astype(bool)
            )
            merged = merged.drop(columns=[state_col])

    fill_zero = [
        "n_storm_events",
        "damage_property_sum",
        "damage_crops_sum",
        "deaths_total",
        "injuries_total",
        "n_fema_declarations",
        "fema_active_days",
    ]
    fill_false = [c for c in merged.columns if c.startswith("had_") or c.endswith("_active")]
    for c in fill_zero:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    for c in fill_false:
        if c in merged.columns:
            merged[c] = merged[c].fillna(False).astype(bool)

    state_map, county_map = build_name_lookups(STORM_SRC)
    merged = merged.merge(state_map, on="STATE_FIPS", how="left")
    merged = merged.merge(county_map, on=["STATE_FIPS", "COUNTY_FIPS"], how="left")

    fallback = fema_county_name_fallback(FEMA_ORIG)
    merged = merged.merge(fallback, on=["STATE_FIPS", "COUNTY_FIPS"], how="left")
    merged["COUNTY"] = merged["COUNTY"].fillna(merged["COUNTY_FALLBACK"])
    merged = merged.drop(columns=["COUNTY_FALLBACK"])

    temp_fallback = temp_county_name_fallback(TEMP_SRC)
    merged = merged.merge(temp_fallback, on=["STATE_FIPS", "COUNTY_FIPS"], how="left")
    merged["COUNTY"] = merged["COUNTY"].fillna(merged["COUNTY_TEMP"])
    merged = merged.drop(columns=["COUNTY_TEMP"])

    front_cols = ["STATE_FIPS", "STATE", "COUNTY_FIPS", "COUNTY", "YEAR_MONTH"]
    merged = merged[front_cols + [c for c in merged.columns if c not in front_cols]]

    merged = merged.sort_values(["STATE_FIPS", "COUNTY_FIPS", "YEAR_MONTH"]).reset_index(drop=True)

    # Compact dtypes for CSV: booleans -> 0/1 int8 (saves ~4 chars per cell),
    # counts -> int (drop the .0 suffix), damages -> int (cent precision is
    # noise at month-county grain). Together these roughly halve the file size.
    for c in merged.columns:
        if merged[c].dtype == bool:
            merged[c] = merged[c].astype("int8")
    for c in ["n_storm_events", "deaths_total", "injuries_total",
              "n_fema_declarations", "fema_active_days"]:
        if c in merged.columns:
            merged[c] = merged[c].round().astype("int32")
    for c in ["damage_property_sum", "damage_crops_sum"]:
        if c in merged.columns:
            merged[c] = merged[c].round().astype("int64")
    for c in ["tmax_f", "tmin_f"]:
        if c in merged.columns:
            merged[c] = merged[c].round().astype("Int16")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)

    size_mb = OUT.stat().st_size / 1e6
    n_storm_only = merged["n_storm_events"].gt(0).sum() - (merged["n_storm_events"].gt(0) & merged["n_fema_declarations"].gt(0)).sum()
    n_fema_only = merged["n_fema_declarations"].gt(0).sum() - (merged["n_storm_events"].gt(0) & merged["n_fema_declarations"].gt(0)).sum()
    n_both = (merged["n_storm_events"].gt(0) & merged["n_fema_declarations"].gt(0)).sum()
    print(f"\nMerged: {len(merged):,} rows, {merged.shape[1]} cols")
    print(f"  storm-only months: {n_storm_only:,}")
    print(f"  FEMA-only months:  {n_fema_only:,}")
    print(f"  both:              {n_both:,}")
    print(f"  unique counties:   {merged.drop_duplicates(['STATE_FIPS','COUNTY_FIPS']).shape[0]:,}")
    print(f"  rows missing STATE name:  {merged['STATE'].isna().sum():,}")
    print(f"  rows missing COUNTY name: {merged['COUNTY'].isna().sum():,}")
    print(f"\nhazard flag firing counts (rows where True):")
    flag_cols = [c for c in merged.columns if c.startswith("had_")]
    for c in sorted(flag_cols):
        print(f"  {c:28s} {int(merged[c].sum()):>8,}")
    print(f"\nSaved -> {OUT.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
