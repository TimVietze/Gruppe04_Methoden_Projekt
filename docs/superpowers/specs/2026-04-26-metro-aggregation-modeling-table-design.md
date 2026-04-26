# Metro-Level Aggregation & Modeling Table — Design

**Date:** 2026-04-26
**Status:** Brainstormed and approved by user; ready for implementation plan.

## Goal

Make `Weather_Features.csv` (county × month) joinable to Zillow ZHVI (metro × month) and produce a single modeling-ready CSV that combines all 36 weather features with all 9 ZHVI variants (bottom/top tier all-homes, mid-tier SFR, mid-tier condo, mid-tier per bedroom-count). The resulting table will be the input to the Streamlit price-prediction model.

## Background

The project predicts US housing prices ~12 months out from weather, FEMA disaster, and (later) economic data, so investors can identify attractive vs. risky regions.

The existing pipeline produces `Weather_Features.csv` keyed on `(STATE_FIPS, COUNTY_FIPS, YEAR_MONTH)` — county-level. Zillow's published ZHVI files are keyed on `(RegionID, month)` where `RegionType="msa"` — metro-level. The two grains do not join, and Zillow does not publish below-metro home values, so an aggregation step is required.

We chose to aggregate the weather UP to metro level (rather than spread Zillow down) so the target retains its true variance and the join is honest.

## Architecture

Two new pipeline stages added after the existing three:

```
EXISTING (renamed with step prefixes)
  1_StormEvents_Aggregation.py            ->  StormEvents_ALL.csv
  2_Fema_DisasterDeclarations_Cleaning.py ->  Fema_DisasterDeclarations_Cleaned.csv
  3_WeatherFeatures_Aggregation.py        ->  Weather_Features.csv  (county x month, 36 cols)

NEW
  4_WeatherFeatures_Metro_Aggregation.py  ->  Weather_Features_Metro.csv  (metro x month, 34 cols)
  5_ModelingTable_Aggregation.py          ->  Modeling_Table.csv          (metro x month, 43 cols)
```

Both new scripts are standalone one-shot data jobs that read inputs from disk and write a single CSV. No tests; sanity prints at the end of each run.

## External inputs (already on disk)

Both files were uploaded by the user to `Methoden Data/X_Original Data/Other Data Original/`:

- **`Census CBSA Delineation File.csv`** — Census 2023 delineation. ~3,200 county-CBSA mappings. Includes both Metropolitan (393) and Micropolitan (~542) Statistical Areas. Real header is on row 3 (skip 2 junk rows).
- **`Census county population.csv`** — Census Population Estimates Program 2010–2016. We will use `POPESTIMATE2016` as the per-county weight. Static weights (relative county sizes within a metro shift slowly enough that 2016 is fine for 2000–2026 weighting).

One additional small file will be created during implementation:

- **`Zillow_RegionID_to_CBSA_overrides.csv`** — manual override map for Zillow regions whose name doesn't match any shortened Census CBSA Title. Expected size: ~10 entries. Format: `RegionID,RegionName,CBSA_CODE,note`. The only top-100 metro requiring override is **Louisville, KY (RegionID maps to CBSA 31140)** — Census uses the unusual `"Louisville/Jefferson County, KY-IN"` form. The remaining ~9 entries are mid-rank (101–500) metros with similar idiosyncratic Census titles. ~30 micropolitan areas at SizeRank > 500 are intentionally left unmapped and will be dropped with a logged warning — they are too small to model meaningfully.

## Stage 4 — `4_WeatherFeatures_Metro_Aggregation.py`

### Inputs
- `Methoden Data/Weather Data/Weather_Features.csv`
- `Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv`
- `Methoden Data/X_Original Data/Other Data Original/Census county population.csv`

### Output
- `Methoden Data/Weather Data/Weather_Features_Metro.csv` — ~280k rows, 34 cols.

### Schema (34 cols)

```
Keys (3):       CBSA_CODE, CBSA_TITLE, YEAR_MONTH
Storm (12):     n_storm_events, damage_property_sum, damage_crops_sum,
                deaths_total, injuries_total,
                had_tornado, had_hurricane, had_flood, had_drought,
                had_heat, had_winter_storm, had_wildfire
FEMA (16):      n_fema_declarations, fema_active_days,
                had_major_disaster, had_emergency, had_fire_mgmt,
                ia_active, ih_active, pa_active, hm_active,
                had_fema_flood, had_fema_hurricane, had_fema_severe_storm,
                had_fema_fire, had_fema_tornado, had_fema_earthquake,
                had_fema_biological
Temperature (3):  tmax_f, tmin_f, precip_in
```

`STATE`, `COUNTY`, `STATE_FIPS`, `COUNTY_FIPS` are dropped — replaced by `CBSA_CODE` (5-digit Census code) and `CBSA_TITLE` (full Census title, e.g. `"Atlanta-Sandy Springs-Alpharetta, GA"`).

### Aggregation rules

Each rule is chosen to preserve the variable's semantic meaning when collapsed across counties of a metro. Three rules cover all 31 feature columns:

| Rule | Columns | Reasoning |
|---|---|---|
| **Sum** | `damage_property_sum`, `damage_crops_sum`, `deaths_total`, `injuries_total`, `n_storm_events`, `n_fema_declarations` | These are additive across geography — the metro's total damage IS the sum of its counties' damages. |
| **OR (max of 0/1)** | All 7 storm `had_*` flags, all 7 FEMA `had_fema_*` flags, `had_major_disaster`, `had_emergency`, `had_fire_mgmt`, `ia_active`, `ih_active`, `pa_active`, `hm_active` | If any county in the metro experienced the hazard, the metro experienced it. |
| **Population-weighted mean** | `tmax_f`, `tmin_f`, `precip_in`, `fema_active_days` | Intensity variables. Weighting by `POPESTIMATE2016` ensures a 5M-population county counts more than a 5,000-population desert county for "metro temperature." |

### Processing steps

1. Load `Weather_Features.csv`.
2. Load Census Delineation File (skip 2 header rows). Filter to Metropolitan + Micropolitan rows. Build crosswalk `(STATE_FIPS, COUNTY_FIPS) -> (CBSA_CODE, CBSA_TITLE)`. State and county FIPS columns must be aligned to `Weather_Features.csv` integer types.
3. Load county population. Filter to county-level rows (`STCOU` populated). Extract `POPESTIMATE2016` as the per-county weight.
4. Left-join crosswalk onto weather table. Drop rows whose county has no CBSA mapping (~1,800 rural counties). Log dropped count.
5. Left-join population onto the result. If any (county, month) row has missing population, **fail loud** — silently substituting 0 or 1 corrupts weighted means.
6. Group by `(CBSA_CODE, YEAR_MONTH)` and apply the rule per column (sum / max / pop-weighted mean).
7. Re-attach `CBSA_TITLE` to each metro row.
8. Round and type-cast for size discipline (matches existing pipeline conventions): damages/counts/casualties → `int`; `had_*` and `*_active` → `int8`; `tmax_f`/`tmin_f` → `int`; `precip_in` → 2 decimals; `fema_active_days` → 1 decimal.
9. Sort by `(CBSA_CODE, YEAR_MONTH)`. Write CSV.

### Sanity prints (end of run)

- Input rows / output rows / dropped-no-cbsa count / dropped-no-population count
- One worked example (e.g. Atlanta CBSA 12060 for one month): contributing county count, unweighted mean `tmax_f`, population-weighted mean `tmax_f`, plus the column sums

## Stage 5 — `5_ModelingTable_Aggregation.py`

### Inputs
- `Methoden Data/Weather Data/Weather_Features_Metro.csv` (Stage 4 output)
- 9 Zillow CSVs from `Methoden Data/X_Original Data/Housing Data Original/` — one per ZHVI variant (after the raw `time_series` file was removed in this design pass)
- `Methoden Data/X_Original Data/Other Data Original/Census CBSA Delineation File.csv`
- `Methoden Data/X_Original Data/Other Data Original/Zillow_RegionID_to_CBSA_overrides.csv`

### Output
- `Methoden Data/Modeling Data/Modeling_Table.csv` — ~270k rows, 43 cols, ~50 MB.

### Schema (43 cols)

```
Keys (3):       CBSA_CODE, CBSA_TITLE, YEAR_MONTH
Weather (31):   inherited from Stage 4 (12 storm + 16 FEMA + 3 temperature)
Housing (9):    zhvi_all_bottom, zhvi_all_top,
                zhvi_sfr_mid, zhvi_condo_mid,
                zhvi_1br_mid, zhvi_2br_mid, zhvi_3br_mid,
                zhvi_4br_mid, zhvi_5br_mid
```

Zillow's `RegionID` and `RegionName` are not retained in the output — `CBSA_CODE` and `CBSA_TITLE` are the canonical keys. The Zillow→CBSA mapping logic stays internal to this stage.

### Zillow → CBSA mapping algorithm

1. Load CBSA Delineation File. Compute a "shortened" form of each `CBSA Title` by extracting the city name before the first `-` and the state abbreviation before the first `-`: e.g. `"Atlanta-Sandy Springs-Alpharetta, GA"` → `"Atlanta, GA"`. Apply Unicode NFKD normalization with ASCII fallback to strip diacritics and mojibake.
2. Build the lookup dict `shortened_title -> CBSA_CODE`.
3. For each Zillow `RegionName`, normalize the same way and look up the CBSA code.
4. Apply the manual overrides file: rows whose `RegionID` appears there bypass the name match and use the override's `CBSA_CODE` directly.
5. Log each unmatched `RegionName`. Drop unmatched rows from downstream processing. Expected unmatched count: ~30 (all SizeRank > 500, all micropolitan, all uneconomic to manually map).

This algorithm achieves **95.5% match rate by name alone** before overrides; with ~10 manual overrides it reaches the practical ceiling of ~96.6% (the ~30 dropped rows are intentional).

### Processing steps

1. Load `Weather_Features_Metro.csv`.
2. Build the Zillow→CBSA mapping (above).
3. For each of the 9 Zillow files: load the wide format, melt month columns into long format `(RegionID, YEAR_MONTH, zhvi_value)`, normalize Zillow's date format `"YYYY-MM-DD"` to `"YYYY-MM"`, attach `CBSA_CODE` via the mapping, drop unmapped rows.
4. Combine the 9 long frames into one wide frame keyed on `(CBSA_CODE, YEAR_MONTH)` with one column per ZHVI variant. If a CBSA maps to multiple Zillow `RegionID`s (rare; possible for cross-state metros where Zillow and Census disagree), take the first non-null value per cell and log the conflict.
5. Inner-join Stage 4's `Weather_Features_Metro` onto this Zillow frame on `(CBSA_CODE, YEAR_MONTH)`. CBSAs with no Zillow data at all are dropped naturally.
6. Filter to `YEAR_MONTH >= "2000-01"`. Pre-2000 Zillow rows have no weather features and are dropped.
7. Sort by `(CBSA_CODE, YEAR_MONTH)`. Write CSV.

### NaN policy

ZHVI cells stay NaN where Zillow has no value (e.g. small metros without enough sales for the 5+ bedroom slice). **No imputation.** The modeling code handles missingness per target — different targets will have different valid row counts.

### Sanity prints

- Per-Zillow-file: rows-loaded, rows-mapped, rows-unmapped (with a sample of unmapped names)
- Final row count
- Non-null count per ZHVI column (so sparsity of `zhvi_5br_mid` etc. is visible)
- One worked example: pick one CBSA × one recent month, print all 43 columns

## Folder layout (after this work)

```
Methoden Data/
    X_Original Data/
        Housing Data Original/
            Zillow_Filename_Tokens.csv          (already created)
            <9 Zillow CSVs>
        Other Data Original/
            Census CBSA Delineation File.csv
            Census county population.csv
            Zillow_RegionID_to_CBSA_overrides.csv  NEW
        Weather Data Original/
            ...
    Weather Data/
        Storm/StormEvents_ALL.csv
        Fema/Fema_DisasterDeclarations_Cleaned.csv
        Weather_Features.csv
        Weather_Features_Metro.csv               NEW (Stage 4 output)
    Modeling Data/                                NEW FOLDER
        Modeling_Table.csv                        NEW (Stage 5 output)

Data Cleaning Scripts/
    1_StormEvents_Aggregation.py
    2_Fema_DisasterDeclarations_Cleaning.py
    3_WeatherFeatures_Aggregation.py
    4_WeatherFeatures_Metro_Aggregation.py        NEW
    5_ModelingTable_Aggregation.py                NEW
```

## Edge cases and design decisions

1. **Counties with no CBSA mapping (~1,800)** — dropped at Stage 4. Zillow does not cover these regions, so they have no target.
2. **Puerto Rico / Guam dual FIPS** — `Weather_Features.csv` contains both NOAA-coded (PR=99, GU=98) and Census-coded (PR=72, GU=66) rows for some territory events. Only the Census-coded rows match the CBSA delineation file. The NOAA-coded twins are dropped at Stage 4. This is consistent with the existing pipeline's caveat that Census codes are the join-correct ones.
3. **Multi-state CBSAs** (e.g. NYC, Kansas City, Louisville) — handled naturally. Counties from each state are summed/averaged into the same CBSA row.
4. **Statewide FEMA Z-flags** — `Weather_Features.csv` already has these flags propagated to every county in the state. OR-aggregating across the metro keeps them statewide-correct.
5. **Counties present in weather but absent from population file** — fail loud. Likely indicates a Connecticut-style FIPS reorganization or a new county that postdates the 2016 population vintage. Adding a manual entry to a population override file is the right fix; silently zeroing is not.
6. **Static population weights** — `POPESTIMATE2016` is used uniformly across 2000–2026. Time-varying weights are out of scope for this project; relative county sizes within a metro shift slowly enough that the bias is negligible.
7. **Micropolitan areas kept** — Zillow lumps Metro and Micropolitan under `RegionType="msa"`. We keep both through Stage 4. The Stage 5 join filters naturally to whatever Zillow covers. Downstream modeling can decide whether to filter to Metropolitan-only via `Metropolitan/Micropolitan Statistical Area` if needed.
8. **Missing all-homes-mid-tier ZHVI** — only `tier_0.0_0.33` (bottom) and `tier_0.67_1.0` (top) all-homes files are SA versions; the mid-tier all-homes file Zillow distributes is the raw `time_series` variant, which mixes badly with the SA variants in modeling. The raw file was deleted from the project; mid-tier targets are still available via the property-type slices (SFR mid, condo mid) and bedroom-count slices.

## Verification approach

Each script's sanity prints (above) are the verification. No formal unit tests — these are one-shot data jobs. The user spot-checks the prints against expectations and reads one worked example per stage. If a print looks off, the relevant step gets re-examined before proceeding.

## Out of scope (deliberately deferred)

- Macro features (CPI, Fed funds rate, mortgage rate). National-only series — easy to add later as a Stage 6 join without rerunning Stage 4 or 5.
- Additional regional sources (USFS wildfire perimeters, USGS earthquakes, EPA AQI). Each can be added as a new feature column in Stage 4 once the data is downloaded.
- An NWS-zone → county crosswalk to fix the Z-flag damage attribution. Existing caveat #1 in the project README.
- Time-varying population weights.
- A formal Zillow-RegionID → CBSA crosswalk file from Zillow Research (would be cleaner than name-matching but requires either Zillow's API or a community-maintained crosswalk; not worth the integration risk for this milestone).
- Any change to the existing scripts beyond renaming for step ordering. The county-level `Weather_Features.csv` is the input to Stage 4 and otherwise untouched.

## Reproduce (after this work lands)

```
cd "FS26_Methoden_Project"
python3 "Data Cleaning Scripts/1_StormEvents_Aggregation.py"
python3 "Data Cleaning Scripts/2_Fema_DisasterDeclarations_Cleaning.py"
python3 "Data Cleaning Scripts/3_WeatherFeatures_Aggregation.py"
python3 "Data Cleaning Scripts/4_WeatherFeatures_Metro_Aggregation.py"
python3 "Data Cleaning Scripts/5_ModelingTable_Aggregation.py"
```

Final modeling-ready table lands in `Methoden Data/Modeling Data/Modeling_Table.csv`.
