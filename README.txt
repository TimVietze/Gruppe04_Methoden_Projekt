================================================================================
FS26 Methoden Project — Data Cleaning & Aggregation
================================================================================

Mini capstone for "Data Science & AI for Business" (FS26, HSG). The goal is a
Streamlit application that predicts US county-level housing prices over the
next ~12 months from past weather, FEMA disaster, and economic data, so
investors can identify counties that look attractive vs. risky.

This README documents the data pipeline that turns the raw downloads under
`Methoden Data/X_Original Data/` into the modeling-ready feature table
`Methoden Data/Weather Data/Weather_Features.csv`.


================================================================================
Data sources (raw)
================================================================================

NOAA Storm Events:
    https://www.ncei.noaa.gov/stormevents/ftp.jsp
    -> per-year CSVs, 2000..2026, in
       Methoden Data/X_Original Data/Weather Data Original/Storm Original/

FEMA Disaster Declarations Summary v2:
    https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2
    -> single CSV (1953..present) in
       Methoden Data/X_Original Data/Weather Data Original/

NOAA nClimDiv (per-county monthly temperature & precipitation):
    https://www.ncei.noaa.gov/pub/data/cirs/climdiv/
    -> processed externally into Temp_per_county_month.csv in
       Methoden Data/X_Original Data/Weather Data Original/

Zillow ZHVI (housing target):
    https://www.zillow.com/research/data/
    -> per-tier CSVs in
       Methoden Data/X_Original Data/Housing Data Original/


================================================================================
Folder layout
================================================================================

Methoden Data/
    X_Original Data/                 immutable raw downloads, never edited
        Housing Data Original/       Zillow ZHVI per tier / bedroom count
        Weather Data Original/
            Storm Original/          27 yearly NOAA Storm Events CSVs
            Fema_DisasterDeclarationsSummaries.csv
            Fema_Disaster_Declarations_Fields.csv  (field dictionary)
            Temp_per_county_month.csv
    Weather Data/                    cleaned outputs of the pipeline
        Storm/StormEvents_ALL.csv
        Fema/Fema_DisasterDeclarations_Cleaned.csv
        Weather_Features.csv         <- the modeling-ready table

Data Cleaning Scripts/
    StormEvents_Aggregation.py
    Fema_DisasterDeclarations_Cleaning.py
    WeatherFeatures_Aggregation.py


================================================================================
Pipeline (run in this order)
================================================================================

1) StormEvents_Aggregation.py
   IN:  27 yearly Storm Original/StormEvents_YYYY.csv files
   OUT: Methoden Data/Weather Data/Storm/StormEvents_ALL.csv  (~78 MB)

   - Combines all year files into one chronological CSV (2000-01 .. 2026-04).
   - Drops the 33 columns we don't need for county-month modeling: the two
     long event narratives (~1 GB of free text), redundant date breakdowns,
     sub-county lat/lon/location strings, admin metadata (WFO, SOURCE, IDs,
     STATE name -- recovered later), and the rarely-populated tornado
     "other county" fields.
   - Drops rows with CZ_TYPE='M' (marine zones, no land housing impact).
   - Drops zero-impact rows (no damage, no injuries, no deaths) for low-signal
     event types: Hail, Thunderstorm Wind, Heavy Snow, Heavy Rain, Dense Fog,
     Lightning, Frost/Freeze, Winter Weather. Tornado / Flood / Hurricane /
     Drought / Heat zero-impact rows are kept -- the occurrence itself is the
     signal for those.
   - Parses DAMAGE_PROPERTY / DAMAGE_CROPS from "2K" / "1.50M" / "100B"
     strings into floats (NaN preserved, not coerced to 0).
   - Builds BEGIN_DATE / END_DATE as YYYY-MM-DD (drops time-of-day, which
     adds nothing at month-county grain).
   - Re-adds STATE and CZ_NAME (county name) so the file isn't only numbers.

   ~1.6M raw rows -> ~910k rows, 51 cols -> 20 cols.

2) Fema_DisasterDeclarations_Cleaning.py
   IN:  Fema_DisasterDeclarationsSummaries.csv  (~70k rows, 28 cols)
   OUT: Methoden Data/Weather Data/Fema/Fema_DisasterDeclarations_Cleaned.csv
        (~3.6 MB, ~50k rows, 14 cols)

   - Keeps only fields useful for a county-month feature pipeline: FIPS join
     keys, the three relevant date columns (incidentBeginDate / End /
     declarationDate), the categorical hazard fields, and the four assistance
     program flags (IA, IH, PA, HM) as severity proxies. Drops admin
     identifiers, free-text titles, and program filing/closeout dates.
   - Zero-pads FIPS codes as strings (so leading zeros survive joins).
   - Adds derived `is_statewide` flag for rows with fipsCountyCode='000' so
     downstream code can decide whether to drop or explode them.
   - Filters to incidents on/after 2000-01-01 to align with Zillow coverage.

3) WeatherFeatures_Aggregation.py
   IN:  StormEvents_ALL.csv,
        Fema_DisasterDeclarations_Cleaned.csv,
        Fema_DisasterDeclarationsSummaries.csv (raw, for designatedArea
            fallback name lookup),
        Temp_per_county_month.csv
   OUT: Methoden Data/Weather Data/Weather_Features.csv  (~99 MB, 36 cols,
        ~1M rows, 3,416 unique counties, 2000-01 .. 2026-04)

   Builds the unified county x month feature table that joins directly to
   Zillow on (STATE_FIPS, COUNTY_FIPS, YEAR_MONTH).

   Storm aggregation:
     - CZ_TYPE='C' rows -> per-county damage / death / injury sums plus
       per-county hazard flags (had_tornado, had_flood, ...).
     - CZ_TYPE='Z' rows -> state-level hazard-flag fallback. Many hazards
       (Hurricane, Drought, Winter Storm, Heat, Wildfire) are reported by
       NOAA at the NWS-zone level, not the county level. Without a
       zone->county crosswalk we can't put their damages in a specific
       county, but we can OR-aggregate the boolean had_* flags per
       (state, month) and then propagate them to every county-month in that
       state. This is why had_hurricane fires on ~18,600 county-months
       instead of just ~2.

   FEMA aggregation:
     - is_statewide rows are dropped.
     - Each declaration is exploded across the months its incident was
       active, so a 6-week event contributes to two months. fema_active_days
       records how many days of the month were under the declaration.
     - n_fema_declarations, had_major_disaster (DR), had_emergency (EM),
       had_fire_mgmt (FM), the four assistance-program flags, and per-
       hazard flags (had_fema_flood, had_fema_hurricane, ...).

   Temperature merge:
     - tmax_f and tmin_f (rounded to integer; 0.5 deg F is noise at month-
       county grain) and precip_in (kept as float, real variability).
     - tavg_f is dropped because it equals (tmax_f + tmin_f) / 2.

   Names:
     - STATE and COUNTY columns are added back so the table is human-readable.
     - 3-tier fallback handles edge cases:
         1. NOAA Storm STATE / CZ_NAME (covers ~98%)
         2. Hardcoded territory map for STATE_FIPS that NOAA codes
            non-standardly (Puerto Rico=72 Census vs 99 NOAA, etc.)
         3. FEMA designatedArea (parenthetical stripped, uppercased)
         4. Temp county_name (suffix stripped: "County", "Parish",
            "Borough", "Census Area", "Municipality", "City and Borough")
     - Final coverage: 0 rows missing STATE, 0 rows missing COUNTY.

   Size discipline (the table is checked into git, so it has to fit under
   GitHub's 100 MiB per-file limit):
     - Booleans encoded as 0/1 int8 (instead of "True"/"False") -- ~85 MB
       saving.
     - Damage / count columns saved as integers (no .0 suffix) -- ~7 MB.
     - Three sparse max_* columns dropped (NaN in 99%+ of rows).
     - tavg_f dropped (derivable).
     - tmax_f / tmin_f rounded to integer.


================================================================================
Output: Weather_Features.csv schema
================================================================================

Keys (3):                 STATE_FIPS, COUNTY_FIPS, YEAR_MONTH (YYYY-MM)
Names (2):                STATE, COUNTY  (uppercase, no suffix)

Storm side (12):
    n_storm_events        count of CZ_TYPE='C' events that month
    damage_property_sum   USD, integer
    damage_crops_sum      USD, integer
    deaths_total          DEATHS_DIRECT + DEATHS_INDIRECT
    injuries_total        INJURIES_DIRECT
    had_tornado, had_hurricane, had_flood, had_drought, had_heat,
    had_winter_storm, had_wildfire   (0/1; OR over C-rows in this county
                                      and Z-rows in this state)

FEMA side (16):
    n_fema_declarations         declarations active this month
    fema_active_days            days of the month under any declaration
    had_major_disaster (DR), had_emergency (EM), had_fire_mgmt (FM)
    ia_active, ih_active, pa_active, hm_active   assistance-program flags
    had_fema_flood, had_fema_hurricane, had_fema_severe_storm,
    had_fema_fire, had_fema_tornado, had_fema_earthquake,
    had_fema_biological

Temperature (3):
    tmax_f, tmin_f       degrees Fahrenheit, integer
    precip_in            inches, float (2 decimals)


================================================================================
Known caveats
================================================================================

1. NWS-zone storm rows (CZ_TYPE='Z', ~half of NOAA's data) currently
   contribute only to boolean had_* flags via a state-level fallback. Their
   damage and death numbers are NOT attributed to specific counties because
   we lack an NWS zone -> county crosswalk. Adding that crosswalk and
   re-running WeatherFeatures_Aggregation.py will make hurricane / drought /
   winter-storm damages show up at the county level too.

2. Total deaths in the storm data (DEATHS_DIRECT + DEATHS_INDIRECT in the
   StormEvents file) match NOAA's separate StormEvents_Fatalities file to
   within 0.3%. The fatalities file was therefore retired from this pipeline
   -- no information loss for the modeling target.

3. FEMA rows with is_statewide=True (fipsCountyCode='000', ~1,100 rows) are
   dropped in WeatherFeatures_Aggregation. They could be exploded across all
   counties of the state instead, but doing so without population/area
   weighting would over-count.

4. FEMA STATE_FIPS uses standard Census codes; NOAA Storm Events uses its
   own non-standard codes for several territories (Puerto Rico = 99 in NOAA
   but 72 in Census, Guam = 98 vs 66, ...). Where the codes diverge they
   show up as separate rows in Weather_Features.csv. Zillow uses Census
   codes, so the Census side is the join-correct one.

5. NOAA data starts 2000-01; that is the cutoff used everywhere in this
   pipeline. Earlier FEMA history is dropped during cleaning.


================================================================================
Reproduce
================================================================================

cd "FS26_Methoden_Project"
python3 "Data Cleaning Scripts/StormEvents_Aggregation.py"
python3 "Data Cleaning Scripts/Fema_DisasterDeclarations_Cleaning.py"
python3 "Data Cleaning Scripts/WeatherFeatures_Aggregation.py"

Outputs land in `Methoden Data/Weather Data/`.
