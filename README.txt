================================================================================
FS26 Methoden Project — ClimateHome
Mini-Capstone "Data Science & AI for Business", University of St. Gallen
================================================================================

Goal
----
Help investors spot attractive vs. risky U.S. metro housing markets by
combining a 6-month price forecast with a 6-month climate-damage risk
forecast. End product: a Streamlit dashboard ("ClimateHome — Investor View").

Pipeline overview
-----------------
    [Raw data]  ->  [Data Cleaning Scripts]  ->  Modeling_Table.csv
                            |
                            v
    [Model Analysis]  ->  trained models + result CSVs
                            |
                            v
    [streamlit_app]   ->  interactive map + metro-level verdicts


================================================================================
1. DATA
================================================================================
Raw inputs (Methoden Data/X_Original Data/)
  - Zillow ZHVI            : 9 housing-price variants per metro & month
                             (bottom/top tier, single-family, condo, 1-5br)
  - NOAA Storm Events      : 27 yearly CSVs (2000-2026) — storms, damages, deaths
  - FEMA Disaster Decl.    : federal disaster declarations since 1953
  - NOAA nClimDiv          : monthly temperature & precipitation per county
  - BLS LAUS               : monthly unemployment per metro
  - Census CBSA / pop.     : metro delineation & county population weights

Cleaned outputs (Methoden Data/)
  - Weather Data/Weather_Features.csv         county x month, 36 cols, ~1M rows
  - Weather Data/Weather_Features_Metro.csv   CBSA x month, 34 cols, ~290k rows
  - Modeling_Table.csv                        the master modeling table
                                              (CBSA x month, 43 cols, ~225k rows,
                                              855 metros, weather + 9 ZHVI cols)
  - Geo Data/CBSA_20m.geojson                 metro boundaries for the map


================================================================================
2. DATA CLEANING (Data Cleaning Scripts/)  -- run in order
================================================================================
  1_StormEvents_Aggregation.py
     Combines 27 yearly NOAA files, drops noise (marine zones, zero-impact
     low-signal events, free-text narratives), parses damage strings
     ("2K"/"1.50M") to floats. 1.6M -> 910k rows.

  2_Fema_DisasterDeclarations_Cleaning.py
     Keeps the FIPS join keys, the 3 date columns, hazard categories, and the
     4 assistance-program flags (IA/IH/PA/HM). Filters to >=2000-01-01.

  3_WeatherFeatures_Aggregation.py
     Joins storm + FEMA + temperature into one county-month feature table.
     County-level damage sums; state-level OR fallback for NWS-zone rows
     (Z-type) covering hurricanes/drought/winter storms. Adds named state/
     county columns via 4-tier fallback.

  4_WeatherFeatures_Metro_Aggregation.py
     Aggregates county -> CBSA using meaningful rules per column:
        sums for counts/damages, OR for hazard flags, population-weighted
        means for temperature/precipitation.

  5_ModelingTable_Aggregation.py
     Maps Zillow regions to Census CBSAs (96.6% match rate via name-shorten
     transform + 6-entry manual override file). Melts the 9 ZHVI variants
     and inner-joins onto the metro weather table.

  6_CBSA_Geodata.py
     Downloads the Census 1:20m CBSA shapefile and filters it to the 855
     CBSAs that appear in the modeling table -> CBSA_20m.geojson (for the map).


================================================================================
3. MODELING (Model Analysis/)
================================================================================
Three independent modeling tracks, each in its own folder with code, trained
.pkl models, and result CSVs/plots.

(A) Prediction % change 6 months/
    First attempt at the actual business question: forecast the 6-month %
    price change from climate + economic features only (no housing prices).
    Result: best model R^2 = -1.8 -> climate/disaster signal alone is not
    enough to predict the % change. Useful negative result; informs why
    Track (B) was added.

(B) Prediction absolute price 6 months/         <-- powers the dashboard
    Predicts the absolute price (USD) 6 months out per (metro, category).
    Pipeline:
      1 Table Adjustment/  filter >= 2015-01, melt 9 ZHVI cols into a long
                           "category" column, drop NaN rows, log-transform
                           prices  -> Modeling_Table_absolute.csv (874k rows)
      2 Modeling/          two experiments share identical code except for
                           the feature list:
                             "prices as feature"     uses log(price_now)
                             "no prices as feature"  drops price_now (stress
                                                     test of non-price signal)
      3 Results/           summary report, model comparison, per-category
                           metrics, best hyperparameters, hold-out predictions,
                           5 pickled best models per experiment, comparison
                           graphs.
    Method:
      - Log-target regression (multiplicative pricing -> additive in log)
      - CBSA fed as feature: one-hot (sparse) for linear, ordinal for trees
      - TimeSeriesSplit(5) + RandomizedSearchCV(n_iter=10) on the train block
      - Hold-out test: rows after 2024-08
      - 5 algorithms compared: LinearRegression, Ridge, RandomForest,
        XGBoost, HistGradientBoosting
    Result:
      - With price_now : Ridge wins with R^2 = 0.998, test RMSE ~ $12k
                         (model essentially learns "price_now x growth rate")
      - Without price  : Ridge again wins with R^2 = 0.86, RMSE ~ $100k
                         (CBSA fixed effects carry most of the remaining signal)
    The "prices as feature" Ridge predictions for 2026-02 -> 2026-08 are the
    price feed for the dashboard (see absolute_predict_2026_02.py).

(C) Prediction Damage Property 6 month/
    Two-stage damage-risk model (climate/disaster features only, no prices):
      1. Classification: will any property damage occur in next 6 months?
         (LogReg / RandomForest / HistGB)
      2. Regression:    if it does, how large in USD?
         (Ridge / RandomForest / HistGB on log_damage_sum_next_6m)
    Cross-validation uses 11 expanding yearly folds (train ends Dec of
    year N, test is the following year). Final per-metro output combines
    P(damage) * E[damage|damage] -> expected_damage_6m, which is then
    percentile-ranked into a business_risk_score (Low / Moderate / High /
    Very High). This is the risk feed for the dashboard.


================================================================================
4. STREAMLIT APP (streamlit_app/)
================================================================================
Loads the two model outputs for the Feb 2026 baseline (forecast window
Mar-Aug 2026) and joins them per metro:

  app.py   entry point, layout, sidebar filters
  data.py  pure pandas loaders + the score join
             price_change_pct      from Ridge predictions (Track B)
             damage_risk_percentile, expected_damage_6m  from Track C
             combined_score = price_change_percentile - risk_percentile
                              (range ~ -100 ... +100)
  viz.py   Plotly choropleth on CBSA_20m.geojson + KPI cards + Top10/
           Bottom10 tables + per-metro detail panel
  text.py  category labels, risk bands, and the rule-based investor verdict
             ("Strong buy", "High reward / high risk", "Avoid", ...)
  tests/   unit tests for data and text modules

User picks a housing category and a map metric (Investor Score / Price
change / Risk percentile). Each metro click expands to current price,
forecast price, expected damage, and the verdict.

Run locally
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py


================================================================================
5. REPRODUCE END-TO-END
================================================================================
    # 1. Clean data
    python3 "Data Cleaning Scripts/1_StormEvents_Aggregation.py"
    python3 "Data Cleaning Scripts/2_Fema_DisasterDeclarations_Cleaning.py"
    python3 "Data Cleaning Scripts/3_WeatherFeatures_Aggregation.py"
    python3 "Data Cleaning Scripts/4_WeatherFeatures_Metro_Aggregation.py"
    python3 "Data Cleaning Scripts/5_ModelingTable_Aggregation.py"
    python3 "Data Cleaning Scripts/6_CBSA_Geodata.py"

    # 2. Train models (output: .pkl + result CSVs)
    python3 "Model Analysis/Prediction absolute price 6 months/1 Table Adjustment/absolute_modeling_table_adjustment.py"
    python3 "Model Analysis/Prediction absolute price 6 months/2 Modeling/prices as feature/absolute_modeling_training_testing.py"
    python3 "Model Analysis/Prediction absolute price 6 months/2 Modeling/prices as feature/absolute_predict_2026_02.py"
    python3 "Model Analysis/Prediction Damage Property 6 month/damage_prediction_6m.py"

    # 3. Launch the dashboard
    streamlit run streamlit_app/app.py


================================================================================
6. REPO LAYOUT
================================================================================
    Methoden Data/                  raw + cleaned data, geojson
    Data Cleaning Scripts/          steps 1 - 6 above
    Model Analysis/                 three modeling tracks (% change,
                                    absolute price, damage risk)
    streamlit_app/                  dashboard (app + data + viz + text + tests)
    ClimateHome_Proposal.pdf        the original project proposal
    docs/                           internal notes (Claude superpowers)
    ClassDocuments/                 (gitignored) lecture materials


================================================================================
7. KNOWN LIMITATIONS
================================================================================
  - NWS-zone storm rows contribute only to boolean flags (no county-level
    crosswalk for hurricane/drought damages yet).
  - The Census population file is CSA-restricted; ~40% of CBSA counties
    fall back to weight=1 in the temperature aggregation.
  - "prices as feature" inflates R^2 because price_now is by far the
    strongest predictor of price_next_6m. The "no prices" experiment is
    the honest test of how much climate/economic signal there is.
  - Track (A) showed that predicting % change from climate-only features
    is not feasible at this horizon -- the dashboard therefore uses the
    absolute-price model and derives % change from it.
  - Several proposal data sources (USFS wildfire perimeters, NOAA heat
    days, USGS earthquakes, EPA AQI) are not yet wired in.
