"""
6_AnnualModelingTable.py

Convert the monthly CBSA-level Modeling_Table.csv into an annual
modeling table suitable for one-year-ahead housing price growth prediction.

Each output row represents one (CBSA_CODE, YEAR=t) observation:
  - Housing features: year-end (December) ZHVI price level and prior-year growth
  - Weather/climate features: full-year aggregates for year t
  - Target: growth from December t to December t+1
    (NaN for 2025 rows, which are used for forward prediction)

No year t+1 values appear as features. The intermediate column
zhvi_sfr_mid_dec_next is used only to compute the target and then dropped.

Input:  Methoden Data/Modeling Data/Modeling_Table.csv
Output: Methoden Data/Modeling Data/Annual_Modeling_Table.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "Methoden Data" / "Modeling Data" / "Modeling_Table.csv"
OUT_FILE = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table.csv"

# ── column groups ──────────────────────────────────────────────────────────────

SUM_COLS = [
    "n_storm_events", "damage_property_sum", "damage_crops_sum",
    "deaths_total", "injuries_total", "n_fema_declarations",
]
MAX_COLS = [
    "had_tornado", "had_hurricane", "had_flood", "had_drought",
    "had_heat", "had_winter_storm", "had_wildfire",
    "had_major_disaster", "had_emergency", "had_fire_mgmt",
    "ia_active", "ih_active", "pa_active", "hm_active",
    "had_fema_flood", "had_fema_hurricane", "had_fema_severe_storm",
    "had_fema_fire", "had_fema_tornado", "had_fema_earthquake",
    "had_fema_biological",
]
MEAN_COLS = ["tmax_f", "tmin_f", "precip_in", "fema_active_days"]

WEATHER_SOURCE_COLS = SUM_COLS + MAX_COLS + MEAN_COLS

HOUSING_FEATURES   = ["zhvi_sfr_mid_dec", "zhvi_sfr_mid_dec_prev", "growth_prev"]
WEATHER_FEATURES   = [c + "_ann" for c in WEATHER_SOURCE_COLS]


def main():
    # ── 1. load and filter ─────────────────────────────────────────────────────
    print(f"Reading {IN_FILE.name} ...")
    mt = pd.read_csv(IN_FILE, dtype={"CBSA_CODE": str})
    print(f"  raw: {len(mt):,} rows, {mt['CBSA_CODE'].nunique()} CBSAs, "
          f"months {mt['YEAR_MONTH'].min()}..{mt['YEAR_MONTH'].max()}")

    # keep 2009+ and drop partial 2026
    mt = mt[(mt["YEAR_MONTH"] >= "2009-01") & (mt["YEAR_MONTH"] < "2026-01")].copy()
    mt["YEAR"] = mt["YEAR_MONTH"].str[:4].astype(int)
    print(f"  after 2009+ / drop-2026 filter: {len(mt):,} rows")

    # ── 2. extract December ZHVI price level per (CBSA, YEAR) ─────────────────
    dec_rows = mt[mt["YEAR_MONTH"].str.endswith("-12")][
        ["CBSA_CODE", "YEAR", "zhvi_sfr_mid"]
    ].copy()
    dec_rows = dec_rows.rename(columns={"zhvi_sfr_mid": "zhvi_sfr_mid_dec"})
    # guard: if a CBSA has duplicate December rows (shouldn't happen), keep first
    dec_rows = dec_rows.drop_duplicates(subset=["CBSA_CODE", "YEAR"])

    # ── 3. aggregate weather/climate features to annual ───────────────────────
    agg_dict = {}
    for c in SUM_COLS:
        agg_dict[c] = "sum"
    for c in MAX_COLS:
        agg_dict[c] = "max"
    for c in MEAN_COLS:
        agg_dict[c] = "mean"

    annual_weather = (
        mt.groupby(["CBSA_CODE", "CBSA_TITLE", "YEAR"])
        .agg(agg_dict)
        .reset_index()
    )
    annual_weather = annual_weather.rename(
        columns={c: c + "_ann" for c in WEATHER_SOURCE_COLS}
    )

    # ── 4. join December ZHVI onto annual weather table ───────────────────────
    annual = annual_weather.merge(dec_rows, on=["CBSA_CODE", "YEAR"], how="left")

    # ── 5. housing history features (computed within CBSA group) ──────────────
    annual = annual.sort_values(["CBSA_CODE", "YEAR"]).reset_index(drop=True)
    grp = annual.groupby("CBSA_CODE")["zhvi_sfr_mid_dec"]

    # prior year-end price
    annual["zhvi_sfr_mid_dec_prev"] = grp.shift(1)
    # prior year's growth rate (the strongest single predictor)
    annual["growth_prev"] = (
        (annual["zhvi_sfr_mid_dec"] - annual["zhvi_sfr_mid_dec_prev"])
        / annual["zhvi_sfr_mid_dec_prev"]
    )

    # ── 6. target variable (forward-shifted; dropped after target column built) ─
    annual["zhvi_sfr_mid_dec_next"] = grp.shift(-1)
    annual["target_growth_1y"] = (
        (annual["zhvi_sfr_mid_dec_next"] - annual["zhvi_sfr_mid_dec"])
        / annual["zhvi_sfr_mid_dec"]
    )
    # remove the intermediate next-year price — must never appear as a feature
    annual = annual.drop(columns=["zhvi_sfr_mid_dec_next"])

    # ── 7. drop rows missing required housing history ─────────────────────────
    before = len(annual)
    annual = annual.dropna(subset=["zhvi_sfr_mid_dec", "zhvi_sfr_mid_dec_prev"]).copy()
    print(f"  dropped {before - len(annual):,} rows with missing Dec ZHVI or prev-year price "
          f"(expected: mostly 2009 rows)")

    # ── 8. assign split label ──────────────────────────────────────────────────
    tgt_null = annual["target_growth_1y"].isna()
    split = np.where(
        annual["YEAR"] == 2025,
        "predict",
        np.where(
            (annual["YEAR"] <= 2021) & ~tgt_null,
            "train",
            np.where(
                (annual["YEAR"] >= 2022) & (annual["YEAR"] <= 2024) & ~tgt_null,
                "test",
                None,
            ),
        ),
    )
    annual["split"] = split
    # drop the small residual where year is in range but target is unexpectedly null
    before = len(annual)
    annual = annual[annual["split"].notna()].copy()
    if before > len(annual):
        print(f"  dropped {before - len(annual):,} rows with unassignable split "
              f"(unexpected null target outside 2025)")

    # ── 9. enforce column order ────────────────────────────────────────────────
    col_order = (
        ["CBSA_CODE", "CBSA_TITLE", "YEAR", "split", "target_growth_1y"]
        + HOUSING_FEATURES
        + WEATHER_FEATURES
    )
    annual = annual[col_order].reset_index(drop=True)

    # ── 10. write output ───────────────────────────────────────────────────────
    annual.to_csv(OUT_FILE, index=False)
    size_mb = OUT_FILE.stat().st_size / 1_048_576

    # ── 11. sanity summary ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SANITY SUMMARY — Annual_Modeling_Table.csv")
    print("=" * 60)
    print(f"Output shape       : {annual.shape[0]:,} rows × {annual.shape[1]} columns")
    print(f"File size          : {size_mb:.2f} MB")
    print(f"Year range         : {annual['YEAR'].min()} – {annual['YEAR'].max()}")
    print(f"Unique CBSAs       : {annual['CBSA_CODE'].nunique()}")
    print()
    print("Rows per split:")
    for label, count in annual["split"].value_counts().sort_index().items():
        years = annual.loc[annual["split"] == label, "YEAR"]
        print(f"  {label:8s}  {count:>5,}  (years {years.min()}–{years.max()})")
    print()
    tgt = annual["target_growth_1y"]
    print(f"Target non-null    : {tgt.notna().sum():,}")
    print(f"Target null        : {tgt.isna().sum():,}  (2025 predict rows, as expected)")
    print()
    nonnull = annual[tgt.notna()]["target_growth_1y"]
    print(f"Target statistics  (train+test rows):")
    print(f"  mean             : {nonnull.mean()*100:+.2f}%")
    print(f"  std              : {nonnull.std()*100:.2f}%")
    print(f"  min              : {nonnull.min()*100:+.2f}%")
    print(f"  max              : {nonnull.max()*100:+.2f}%")
    print(f"  pct positive     : {(nonnull > 0).mean()*100:.1f}%")
    print()
    print(f"Housing features   ({len(HOUSING_FEATURES)}):")
    for f in HOUSING_FEATURES:
        nn = annual[f].notna().sum()
        print(f"  {f:35s}  {nn:,} non-null")
    print()
    print(f"Weather/climate features  ({len(WEATHER_FEATURES)}):")
    for f in WEATHER_FEATURES:
        nn = annual[f].notna().sum()
        print(f"  {f:35s}  {nn:,} non-null")
    print()
    print("First 5 rows (key columns):")
    preview_cols = ["CBSA_CODE", "CBSA_TITLE", "YEAR", "split",
                    "target_growth_1y", "zhvi_sfr_mid_dec", "growth_prev",
                    "n_storm_events_ann", "had_major_disaster_ann", "tmax_f_ann"]
    print(annual[preview_cols].head().to_string(index=False))
    print()
    print(f"Wrote: {OUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
