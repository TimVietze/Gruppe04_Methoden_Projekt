# Metro Aggregation & Modeling Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two new pipeline stages that aggregate `Weather_Features.csv` from county-level to metro-level (CBSA) and then join 9 Zillow ZHVI variants to produce a single modeling-ready CSV.

**Architecture:** Two standalone Python scripts in `Data Cleaning Scripts/`. Each is a one-shot data job: reads inputs, transforms, writes a CSV, prints sanity stats. No tests — verification is via the sanity prints, matching the existing pipeline's pattern (see spec for rationale).

**Tech Stack:** Python 3 + pandas. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-26-metro-aggregation-modeling-table-design.md`

---

## File Structure

**Will create:**
- `Data Cleaning Scripts/4_WeatherFeatures_Metro_Aggregation.py` — Stage 4 script
- `Data Cleaning Scripts/5_ModelingTable_Aggregation.py` — Stage 5 script
- `Methoden Data/X_Original Data/Other Data Original/Zillow_RegionID_to_CBSA_overrides.csv` — manual mapping for unmatched Zillow regions
- `Methoden Data/Modeling Data/` — new folder
- `Methoden Data/Weather Data/Weather_Features_Metro.csv` — Stage 4 output
- `Methoden Data/Modeling Data/Modeling_Table.csv` — Stage 5 output

**Will modify:**
- `README.txt` — extend "Pipeline" section with Stages 4 & 5, extend "Folder layout" and "Reproduce" sections, add new caveats.

---

### Task 1: Build the Zillow→CBSA overrides CSV

The override file feeds Stage 5. I'll generate it programmatically by inspecting which Zillow `RegionName`s fail the name-match algorithm, then look each one up in the Census Delineation File and pick the right CBSA code based on city/state proximity. Only entries with rank ≤ 500 get an override; rank > 500 get dropped.

**Files:**
- Create: `Methoden Data/X_Original Data/Other Data Original/Zillow_RegionID_to_CBSA_overrides.csv`

- [ ] **Step 1.1: Run discovery script** to list each unmatched Zillow region (rank ≤ 500) alongside Census candidates whose city prefix matches.

- [ ] **Step 1.2: Hand-pick the right CBSA code** for each candidate based on state and metropolitan area composition.

- [ ] **Step 1.3: Write the override CSV** with columns `RegionID,RegionName,CBSA_CODE,note`.

---

### Task 2: Write `4_WeatherFeatures_Metro_Aggregation.py`

**Files:**
- Create: `Data Cleaning Scripts/4_WeatherFeatures_Metro_Aggregation.py`

The script:
1. Loads `Weather_Features.csv`, the CBSA Delineation File (skip 2 header rows, filter to Metro+Micro), and the county population file.
2. Joins the county→CBSA crosswalk onto weather rows.
3. Joins `POPESTIMATE2016` onto weather rows. Fails loud if any (county, month) lacks population.
4. Groups by `(CBSA_CODE, YEAR_MONTH)` and applies:
   - SUM for `n_storm_events`, `damage_property_sum`, `damage_crops_sum`, `deaths_total`, `injuries_total`, `n_fema_declarations`
   - MAX for all `had_*` and `*_active` flags
   - Population-weighted mean for `tmax_f`, `tmin_f`, `precip_in`, `fema_active_days`
5. Re-attaches `CBSA_TITLE`. Casts and rounds for size discipline.
6. Sorts by `(CBSA_CODE, YEAR_MONTH)`. Writes CSV.
7. Prints sanity stats: input rows, output rows, dropped-no-cbsa, dropped-no-population, one worked example.

- [ ] **Step 2.1:** Implement and write the script.
- [ ] **Step 2.2:** Run it. Verify sanity prints look right. Verify output CSV exists and has the expected schema (3 + 12 + 16 + 3 = 34 cols).

---

### Task 3: Write `5_ModelingTable_Aggregation.py`

**Files:**
- Create: `Data Cleaning Scripts/5_ModelingTable_Aggregation.py`

The script:
1. Loads `Weather_Features_Metro.csv`, the CBSA Delineation File, and the override file.
2. Builds the Zillow→CBSA mapping using:
   - "Shorten" each Census CBSA Title to `<CityPrefix>, <StateAbbrev>` form
   - Unicode NFKD-normalize both sides
   - Apply override file (RegionID-keyed, takes precedence over name match)
3. For each of the 9 Zillow files:
   - Read wide format
   - Melt month columns into long: `(RegionID, YEAR_MONTH, zhvi_value)`
   - Normalize Zillow's "YYYY-MM-DD" to "YYYY-MM"
   - Attach `CBSA_CODE` via the mapping; drop unmapped rows; log unmapped names
4. Combines all 9 long frames into one wide frame keyed `(CBSA_CODE, YEAR_MONTH)` with 9 ZHVI columns.
5. Inner-joins Stage 4's metro weather onto this frame.
6. Filters to `YEAR_MONTH >= "2000-01"`.
7. Sorts by `(CBSA_CODE, YEAR_MONTH)`. Writes CSV.
8. Prints sanity stats: per-file mapping rates, final row count, non-null count per ZHVI column, one worked example.

- [ ] **Step 3.1:** Implement and write the script.
- [ ] **Step 3.2:** Run it. Verify sanity prints look right. Verify output CSV exists, has 43 cols, sizes ~50 MB.

---

### Task 4: Update `README.txt`

**Files:**
- Modify: `README.txt`

Extensions:
- Folder layout: add `Modeling Data/Modeling_Table.csv`, the new override file, the existing `Zillow_Filename_Tokens.csv`, `Weather_Features_Metro.csv`, and the two new scripts (with `4_` and `5_` prefixes)
- Pipeline section: add "4) 4_WeatherFeatures_Metro_Aggregation.py" and "5) 5_ModelingTable_Aggregation.py" stages with input/output and processing notes
- New "Output: Weather_Features_Metro.csv schema" and "Output: Modeling_Table.csv schema" sections
- Known caveats: add a note about the Zillow→CBSA name-match approach and the dropped micropolitan areas
- Reproduce: add the two new `python3` invocation lines

- [ ] **Step 4.1:** Apply edits.

---

### Task 5: Wrap-up verification

- [ ] **Step 5.1:** Confirm both new CSVs exist and have the expected row/column counts.
- [ ] **Step 5.2:** Run a small spot-check query: pick Atlanta CBSA `12060` for `2024-09` (Helene month) and print the row from `Modeling_Table.csv` to confirm hurricane flags + ZHVI prices are present and sane.
- [ ] **Step 5.3:** Print final summary: file paths, sizes, row counts, anything notable from the sanity prints.
