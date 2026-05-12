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

Census CBSA Delineation File / County Population:
    https://www.census.gov/newsroom/press-kits/2017/20170323_popestimates.html
    -> Population

    https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html
    -> Delineation File

================================================================================
Folder layout
================================================================================

Methoden Data/
    X_Original Data/                 immutable raw downloads, never edited
        Housing Data Original/       Zillow ZHVI per tier / bedroom count
            Zillow_Filename_Tokens.csv  (dictionary of filename tokens)
        Weather Data Original/
            Storm Original/          27 yearly NOAA Storm Events CSVs
            Fema_DisasterDeclarationsSummaries.csv
            Fema_Disaster_Declarations_Fields.csv  (field dictionary)
            Temp_per_county_month.csv
        Other Data Original/
            Census CBSA Delineation File.csv
            Census county population.csv
            Zillow_RegionID_to_CBSA_overrides.csv  (manual map for unmatchable Zillow regions)
    Weather Data/                    cleaned outputs of the pipeline
        Storm/StormEvents_ALL.csv
        Fema/Fema_DisasterDeclarations_Cleaned.csv
        Weather_Features.csv         <- the county-level feature table
        Weather_Features_Metro.csv   <- the metro-level feature table (output of stage 4)
    Modeling Data/                   final modeling-ready outputs
        Modeling_Table.csv           <- weather + 9 ZHVI variants joined on (CBSA, month)

Data Cleaning Scripts/
    1_StormEvents_Aggregation.py
    2_Fema_DisasterDeclarations_Cleaning.py
    3_WeatherFeatures_Aggregation.py
    4_WeatherFeatures_Metro_Aggregation.py
    5_ModelingTable_Aggregation.py


================================================================================
Pipeline (run in this order)
================================================================================

1) 1_StormEvents_Aggregation.py
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

2) 2_Fema_DisasterDeclarations_Cleaning.py
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

3) 3_WeatherFeatures_Aggregation.py
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

4) 4_WeatherFeatures_Metro_Aggregation.py
   IN:  Weather_Features.csv,
        Census CBSA Delineation File.csv,
        Census county population.csv
   OUT: Methoden Data/Weather Data/Weather_Features_Metro.csv
        (~28 MB, ~290k rows, 34 cols, ~930 CBSAs)

   Aggregates the county-level Weather_Features to metro level (CBSA), so the
   table can join Zillow's metro-level ZHVI files. Each column gets its own
   aggregation rule based on what the variable means:

     - Sum: n_storm_events, damage_property_sum, damage_crops_sum,
       deaths_total, injuries_total, n_fema_declarations  (additive across
       counties of a metro).
     - Max (OR): every had_* and *_active 0/1 flag (16 cols)  (if any county
       in the metro experienced the hazard, the metro experienced it).
     - Population-weighted mean: tmax_f, tmin_f, precip_in, fema_active_days
       (intensity variables; weighted by Census POPESTIMATE2016 so a 5M
       county counts more than a 5,000 desert county).

   Counties without a CBSA mapping (rural, ~1,500) are dropped -- Zillow has
   no price for them. The uploaded Census county population file only covers
   counties inside Combined Statistical Areas (~60% of CBSA counties), so
   counties without population fall back to weight=1; they then contribute
   negligibly to weighted means while their sum/max contributions stay intact.

5) 5_ModelingTable_Aggregation.py
   IN:  Weather_Features_Metro.csv,
        9 Zillow ZHVI CSVs in Housing Data Original/,
        Census CBSA Delineation File.csv,
        Zillow_RegionID_to_CBSA_overrides.csv
   OUT: Methoden Data/Modeling Data/Modeling_Table.csv
        (~53 MB, ~225k rows, 43 cols, ~855 CBSAs)

   Joins all 9 Zillow ZHVI variants onto the metro weather table, producing
   the final modeling-ready file.

   Zillow regions are mapped to Census CBSA codes by:
     1. Shortening each Census CBSA Title from "Atlanta-Sandy
        Springs-Roswell, GA" to "Atlanta, GA" (city prefix + state),
     2. Unicode-normalizing both sides,
     3. Looking up Zillow's RegionName in the resulting dictionary,
     4. Falling back to a 6-line manual override file for cases the
        shortener cannot reach (Louisville, The Villages, Ogdensburg,
        London KY, California MD, Glenwood Springs).

   Match rate: 96.6% (864 of 894 Zillow MSA rows). The remaining ~30
   unmatched rows are SizeRank > 500 micropolitan areas Zillow lists but
   Census assigns to different parent CBSAs already covered by separate
   Zillow rows -- mapping them would create duplicate keys, so they are
   intentionally dropped.

   Each Zillow file is melted from wide format (one column per month) to
   long, attached to its CBSA_CODE, and combined into a single wide frame
   keyed (CBSA_CODE, YEAR_MONTH) with one column per ZHVI variant. The
   weather table is then inner-joined onto it. Pre-2000 Zillow rows are
   dropped (no weather features to pair with). ZHVI cells with no value
   stay NaN -- not all metros have data for all variants (small metros
   often miss the 5+ bedroom slice).


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
Output: Weather_Features_Metro.csv schema  (Stage 4)
================================================================================

Keys (3):                 CBSA_CODE, CBSA_TITLE, YEAR_MONTH (YYYY-MM)

The 31 feature columns are inherited from Weather_Features.csv but aggregated
to CBSA-month grain (sum / OR / pop-weighted mean -- see stage 4 in the
Pipeline section above).

STATE, COUNTY, STATE_FIPS, COUNTY_FIPS are dropped -- replaced by CBSA_CODE
(5-digit Census code) and CBSA_TITLE (full Census title, e.g.
"Atlanta-Sandy Springs-Roswell, GA").


================================================================================
Output: Modeling_Table.csv schema  (Stage 5)
================================================================================

Keys (3):                 CBSA_CODE, CBSA_TITLE, YEAR_MONTH

Weather (31):             all 31 feature columns from Weather_Features_Metro

Housing (9):              one column per Zillow ZHVI variant
    zhvi_all_bottom         all-homes bottom-tier  (~5th-35th percentile)
    zhvi_all_top            all-homes top-tier     (~65th-95th percentile)
    zhvi_sfr_mid            single-family mid-tier
    zhvi_condo_mid          condos mid-tier
    zhvi_1br_mid ..         mid-tier homes broken out by bedroom count
    zhvi_5br_mid              (1, 2, 3, 4, 5+ bedrooms)

ZHVI cells stay NaN where Zillow has no value for that (metro, month) -- not
all metros have data for all variants. The modeling code is responsible for
handling missingness per target.

See Methoden Data/X_Original Data/Housing Data Original/Zillow_Filename_Tokens.csv
for full documentation of the Zillow filename tokens (uc, sfrcondo, sfr,
condo, tier, sm_sa, etc).


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

6. The uploaded Census county population file is a CSA-restricted subset of
   the Census Population Estimates Program. It only contains counties that
   are part of Combined Statistical Areas, leaving ~40% of CBSA counties
   without a population weight. Stage 4 falls back to weight=1 for those,
   which means they contribute negligibly to pop-weighted temperature /
   precipitation but their sums (damage, deaths, event counts) and OR
   flags are unaffected. Replacing the file with a full county Population
   Estimates file would tighten the temperature weighting; not a blocker.

7. Zillow region names are not directly Census CBSA codes. Stage 5 uses a
   name-shorten transform plus a small (6-entry) manual override file to
   map Zillow RegionID -> CBSA_CODE. Achieves 96.6% match. Roughly 30
   tiny micropolitan areas (SizeRank > 500) Zillow lists are intentionally
   dropped because their Census parent CBSA already has a separate Zillow
   row -- mapping them would corrupt the modeling table.

8. The all-homes mid-tier ZHVI is missing from the project. Zillow only
   distributes that variant in non-seasonally-adjusted form, which mixes
   poorly with the SA variants used everywhere else. The mid-tier is still
   available indirectly via the property-type slices (sfr_mid, condo_mid)
   and the bedroom-count slices (1br_mid through 5br_mid).


================================================================================
Reproduce
================================================================================

cd "FS26_Methoden_Project"
python3 "Data Cleaning Scripts/1_StormEvents_Aggregation.py"
python3 "Data Cleaning Scripts/2_Fema_DisasterDeclarations_Cleaning.py"
python3 "Data Cleaning Scripts/3_WeatherFeatures_Aggregation.py"
python3 "Data Cleaning Scripts/4_WeatherFeatures_Metro_Aggregation.py"
python3 "Data Cleaning Scripts/5_ModelingTable_Aggregation.py"

Outputs land in `Methoden Data/Weather Data/` (stages 1-4) and
`Methoden Data/Modeling Data/` (stage 5).
