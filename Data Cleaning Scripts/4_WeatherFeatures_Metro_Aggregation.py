"""
4_WeatherFeatures_Metro_Aggregation.py

Aggregate the county-level Weather_Features.csv up to metro level (CBSA).
Output joins directly to the metro-level Zillow ZHVI files in stage 5.

Aggregation rules (chosen to preserve each variable's semantic meaning):
  Sum         : n_storm_events, damage_property_sum, damage_crops_sum,
                deaths_total, injuries_total, n_fema_declarations
  Max  (OR)   : every had_* and *_active 0/1 flag (16 columns)
  Pop-mean    : tmax_f, tmin_f, precip_in, fema_active_days
                weighted by Census POPESTIMATE2016

Counties without a CBSA mapping (rural, ~1,800) are dropped.
A county appearing in the weather table but NOT in the population file
fails the run loud -- silently substituting 0 or 1 corrupts weighted means.

Inputs:
  - Methoden Data/Weather Data/Weather_Features.csv
  - Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv
  - Methoden Data/X_Original Data/Other Data Original/Census county population.csv
Output:
  - Methoden Data/Weather Data/Weather_Features_Metro.csv  (~280k rows, 34 cols)
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEATHER_IN = ROOT / "Methoden Data/Weather Data/Weather_Features.csv"
DELIN_IN   = ROOT / "Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv"
POP_IN     = ROOT / "Methoden Data/X_Original Data/Other Data Original/Census county population.csv"
OUT_FILE   = ROOT / "Methoden Data/Weather Data/Weather_Features_Metro.csv"

SUM_COLS = [
    "n_storm_events", "damage_property_sum", "damage_crops_sum",
    "deaths_total", "injuries_total", "n_fema_declarations",
]
MAX_COLS = [
    "had_tornado", "had_hurricane", "had_flood", "had_drought", "had_heat",
    "had_winter_storm", "had_wildfire",
    "had_major_disaster", "had_emergency", "had_fire_mgmt",
    "ia_active", "ih_active", "pa_active", "hm_active",
    "had_fema_flood", "had_fema_hurricane", "had_fema_severe_storm",
    "had_fema_fire", "had_fema_tornado", "had_fema_earthquake", "had_fema_biological",
]
WMEAN_COLS = ["tmax_f", "tmin_f", "precip_in", "fema_active_days"]

OUTPUT_COL_ORDER = (
    ["CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH"]
    + ["n_storm_events", "damage_property_sum", "damage_crops_sum",
       "deaths_total", "injuries_total"]
    + ["had_tornado", "had_hurricane", "had_flood", "had_drought",
       "had_heat", "had_winter_storm", "had_wildfire"]
    + ["n_fema_declarations", "fema_active_days",
       "had_major_disaster", "had_emergency", "had_fire_mgmt",
       "ia_active", "ih_active", "pa_active", "hm_active",
       "had_fema_flood", "had_fema_hurricane", "had_fema_severe_storm",
       "had_fema_fire", "had_fema_tornado", "had_fema_earthquake",
       "had_fema_biological"]
    + ["tmax_f", "tmin_f", "precip_in"]
)


def load_crosswalk():
    delin = pd.read_csv(DELIN_IN, skiprows=2, dtype=str)
    keep = delin["Metropolitan/Micropolitan Statistical Area"].isin([
        "Metropolitan Statistical Area",
        "Micropolitan Statistical Area",
    ])
    delin = delin[keep & delin["FIPS State Code"].notna() & delin["FIPS County Code"].notna()].copy()
    delin["STATE_FIPS"] = delin["FIPS State Code"].astype(int)
    delin["COUNTY_FIPS"] = delin["FIPS County Code"].astype(int)
    cw = delin[["STATE_FIPS", "COUNTY_FIPS", "CBSA Code", "CBSA Title"]].rename(
        columns={"CBSA Code": "CBSA_CODE", "CBSA Title": "CBSA_TITLE"}
    ).drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"])
    return cw


def load_population():
    # The file lost leading zeros in STCOU (Alabama '01007' became '1007') and
    # uses a non-UTF-8 encoding for some rows. Zero-pad to 5 digits before
    # parsing the state/county split.
    try:
        pop = pd.read_csv(POP_IN, dtype=str)
    except UnicodeDecodeError:
        pop = pd.read_csv(POP_IN, dtype=str, encoding="latin-1")
    pop = pop[pop["LSAD"] == "County or equivalent"].copy()
    pop["STCOU_padded"] = pop["STCOU"].str.zfill(5)
    pop["STATE_FIPS"] = pop["STCOU_padded"].str[:2].astype(int)
    pop["COUNTY_FIPS"] = pop["STCOU_padded"].str[2:].astype(int)
    pop["pop_weight"] = pd.to_numeric(pop["POPESTIMATE2016"], errors="coerce")
    pop = pop.dropna(subset=["pop_weight"])
    pop = pop.drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"])
    return pop[["STATE_FIPS", "COUNTY_FIPS", "pop_weight"]]


def main():
    print(f"[Stage 4] reading {WEATHER_IN.name} ...")
    df = pd.read_csv(WEATHER_IN)
    rows_in = len(df)
    counties_in = df.drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"]).shape[0]
    print(f"  loaded {rows_in:,} rows, {counties_in:,} unique counties")

    cw = load_crosswalk()
    print(f"  crosswalk: {len(cw):,} county->CBSA pairs across {cw['CBSA_CODE'].nunique():,} CBSAs")

    pop = load_population()
    print(f"  population: {len(pop):,} county weights from POPESTIMATE2016")

    df = df.merge(cw, on=["STATE_FIPS", "COUNTY_FIPS"], how="left")
    no_cbsa = df["CBSA_CODE"].isna()
    counties_dropped = df.loc[no_cbsa].drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"]).shape[0]
    print(f"  dropped {no_cbsa.sum():,} rows ({counties_dropped:,} counties) -- no CBSA mapping")
    df = df[~no_cbsa].copy()

    df = df.merge(pop, on=["STATE_FIPS", "COUNTY_FIPS"], how="left")
    no_pop = df["pop_weight"].isna()
    if no_pop.any():
        # The uploaded population file is a CSA-restricted subset of the Census
        # Population Estimates and is missing roughly 40% of CBSA counties.
        # Falling back to weight=1 lets the pipeline continue. Counties with
        # weight=1 effectively get excluded from temperature weighting because
        # any other county in the same metro will have weight in the thousands+.
        missing_counties = df.loc[no_pop].drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"]).shape[0]
        total_counties = df.drop_duplicates(["STATE_FIPS", "COUNTY_FIPS"]).shape[0]
        print(f"  WARNING: {missing_counties:,} of {total_counties:,} mapped counties "
              f"({100*missing_counties/total_counties:.1f}%) have no POPESTIMATE2016. "
              f"Falling back to weight=1 -- they contribute negligibly to weighted means.")
        df.loc[no_pop, "pop_weight"] = 1.0

    print(f"  aggregating {len(df):,} county-month rows -> CBSA-month rows ...")

    for c in WMEAN_COLS:
        df[f"_w_{c}"] = df[c].astype(float) * df["pop_weight"]

    agg_dict = {c: "sum" for c in SUM_COLS}
    agg_dict.update({c: "max" for c in MAX_COLS})
    agg_dict.update({f"_w_{c}": "sum" for c in WMEAN_COLS})
    agg_dict["pop_weight"] = "sum"

    out = df.groupby(["CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH"], sort=False).agg(agg_dict).reset_index()

    for c in WMEAN_COLS:
        out[c] = out[f"_w_{c}"] / out["pop_weight"]
        out.drop(columns=[f"_w_{c}"], inplace=True)
    out.drop(columns=["pop_weight"], inplace=True)

    for c in MAX_COLS:
        out[c] = out[c].astype("int8")
    for c in ["n_storm_events", "damage_property_sum", "damage_crops_sum",
              "deaths_total", "injuries_total", "n_fema_declarations"]:
        out[c] = out[c].round().astype("int64")
    out["tmax_f"] = out["tmax_f"].round().astype("int64")
    out["tmin_f"] = out["tmin_f"].round().astype("int64")
    out["precip_in"] = out["precip_in"].round(2)
    out["fema_active_days"] = out["fema_active_days"].round(1)

    out = out[OUTPUT_COL_ORDER]
    out = out.sort_values(["CBSA_CODE", "YEAR_MONTH"]).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"  wrote {OUT_FILE.name} -- {len(out):,} rows, {len(out.columns)} cols, {size_mb:.1f} MB")

    print("\n[Worked example] Atlanta CBSA 12060, 2024-09 (Hurricane Helene month):")
    atl = out[(out["CBSA_CODE"] == "12060") & (out["YEAR_MONTH"] == "2024-09")]
    if len(atl):
        for col, val in atl.iloc[0].items():
            print(f"  {col:30s}  {val}")
        src = df[(df["CBSA_CODE"] == "12060") & (df["YEAR_MONTH"] == "2024-09")]
        unweighted = src["tmax_f"].mean()
        weighted = (src["tmax_f"] * src["pop_weight"]).sum() / src["pop_weight"].sum()
        print(f"\n  contributing counties: {len(src)}")
        print(f"  unweighted mean tmax_f: {unweighted:.2f}")
        print(f"  pop-weighted mean tmax_f: {weighted:.2f}")
    else:
        print("  (no row found -- spot-check skipped)")

    print(f"\n[Stage 4] done.")


if __name__ == "__main__":
    main()
