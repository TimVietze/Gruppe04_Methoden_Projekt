"""
Disaster Impact Model: Price Deviation After Natural Disasters
=============================================================
Research Question: Do natural disasters cause abnormal price movements
in affected housing markets, and which disaster characteristics predict
the magnitude of price impact?

Key design choices:
- Target: Abnormal return (local 6m price change - national 6m mean)
- Focus: Only disaster-affected CBSA-months (at least 1 storm event or FEMA declaration)
- Controls: Pre-disaster price trend, unemployment, seasonality
- Temporal split: train < 2022, test >= 2022
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("DISASTER IMPACT MODEL")
print("Do natural disasters cause abnormal housing price movements?")
print("=" * 80)

df = pd.read_csv("Methoden Data/Modeling_Table.csv")
df['YEAR_MONTH'] = pd.to_datetime(df['YEAR_MONTH'])
df = df.sort_values(['CBSA_CODE', 'YEAR_MONTH']).reset_index(drop=True)
print(f"\nLoaded: {len(df):,} rows, {df['CBSA_CODE'].nunique()} CBSAs, {df['YEAR_MONTH'].min().date()} to {df['YEAR_MONTH'].max().date()}")

# ============================================================================
# 2. CONSTRUCT TARGET: ABNORMAL RETURN
# ============================================================================

print("\n" + "-" * 60)
print("CONSTRUCTING TARGET VARIABLE")
print("-" * 60)

# 6-month forward price change (%)
df['zhvi_fwd6'] = df.groupby('CBSA_CODE')['zhvi_sfr_mid'].shift(-6)
df['pct_change_6m'] = (df['zhvi_fwd6'] - df['zhvi_sfr_mid']) / df['zhvi_sfr_mid'] * 100

# National average price change per month (macro trend control)
df['national_mean_6m'] = df.groupby('YEAR_MONTH')['pct_change_6m'].transform('mean')

# Abnormal return = local price change - national mean
# This isolates the LOCAL deviation from the macro housing cycle
df['abnormal_return_6m'] = df['pct_change_6m'] - df['national_mean_6m']

print(f"Abnormal return stats (all rows with data):")
print(f"  Mean:   {df['abnormal_return_6m'].mean():+.4f}%")
print(f"  Median: {df['abnormal_return_6m'].median():+.4f}%")
print(f"  Std:    {df['abnormal_return_6m'].std():.4f}%")
print(f"  Range:  [{df['abnormal_return_6m'].min():.2f}%, {df['abnormal_return_6m'].max():.2f}%]")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n" + "-" * 60)
print("FEATURE ENGINEERING")
print("-" * 60)

# --- Disaster severity features ---
df['log_damage_property'] = np.log1p(df['damage_property_sum'])
df['log_damage_crops'] = np.log1p(df['damage_crops_sum'])
df['log_damage_total'] = np.log1p(df['damage_property_sum'] + df['damage_crops_sum'])
df['has_casualties'] = ((df['deaths_total'] > 0) | (df['injuries_total'] > 0)).astype(int)
df['casualty_score'] = df['deaths_total'] * 5 + df['injuries_total']

# --- Rolling/cumulative disaster exposure (recent disaster history) ---
for window in [3, 6, 12]:
    df[f'damage_roll{window}m'] = df.groupby('CBSA_CODE')['damage_property_sum'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    df[f'log_damage_roll{window}m'] = np.log1p(df[f'damage_roll{window}m'])
    
    df[f'storms_roll{window}m'] = df.groupby('CBSA_CODE')['n_storm_events'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    
    df[f'fema_days_roll{window}m'] = df.groupby('CBSA_CODE')['fema_active_days'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )
    
    df[f'fema_decl_roll{window}m'] = df.groupby('CBSA_CODE')['n_fema_declarations'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

# --- Disaster type interactions ---
df['n_disaster_types'] = (
    df['had_tornado'] + df['had_hurricane'] + df['had_flood'] + 
    df['had_drought'] + df['had_heat'] + df['had_winter_storm'] + df['had_wildfire']
)
df['is_hurricane_or_flood'] = ((df['had_hurricane'] == 1) | (df['had_flood'] == 1)).astype(int)
df['is_supply_destroyer'] = ((df['had_hurricane'] == 1) | (df['had_wildfire'] == 1) | (df['had_tornado'] == 1)).astype(int)

# --- FEMA response intensity ---
df['fema_response_breadth'] = df['ia_active'] + df['ih_active'] + df['pa_active'] + df['hm_active']
df['fema_program_count'] = (
    df[['ia_active','ih_active','pa_active','hm_active']].sum(axis=1)
)

# --- Pre-disaster price context (control variables) ---
for lag in [3, 6, 12]:
    df[f'zhvi_lag{lag}'] = df.groupby('CBSA_CODE')['zhvi_sfr_mid'].shift(lag)
    df[f'pct_pre{lag}m'] = (df['zhvi_sfr_mid'] - df[f'zhvi_lag{lag}']) / df[f'zhvi_lag{lag}'] * 100

# Price level relative to national median (expensive vs cheap markets)
df['national_median_price'] = df.groupby('YEAR_MONTH')['zhvi_sfr_mid'].transform('median')
df['price_ratio_national'] = df['zhvi_sfr_mid'] / df['national_median_price']
df['log_price_level'] = np.log1p(df['zhvi_sfr_mid'])

# --- Seasonality ---
df['month'] = df['YEAR_MONTH'].dt.month
df['quarter'] = df['YEAR_MONTH'].dt.quarter

# --- Weather context ---
df['temp_range'] = df['tmax_f'] - df['tmin_f']
df['is_extreme_heat'] = (df['tmax_f'] > 95).astype(int)
df['is_extreme_cold'] = (df['tmin_f'] < 10).astype(int)
df['heavy_precip'] = (df['precip_in'] > df['precip_in'].quantile(0.90)).astype(int)

# ============================================================================
# 4. DEFINE FEATURE SET AND FILTER TO DISASTER-AFFECTED ROWS
# ============================================================================

target_col = 'abnormal_return_6m'

feature_cols = [
    # Disaster severity (current month)
    'n_storm_events', 'log_damage_property', 'log_damage_crops', 'log_damage_total',
    'deaths_total', 'injuries_total', 'has_casualties', 'casualty_score',
    
    # Disaster type indicators
    'had_tornado', 'had_hurricane', 'had_flood', 'had_drought',
    'had_heat', 'had_winter_storm', 'had_wildfire',
    'n_disaster_types', 'is_hurricane_or_flood', 'is_supply_destroyer',
    
    # FEMA response
    'n_fema_declarations', 'fema_active_days',
    'had_major_disaster', 'had_emergency', 'had_fire_mgmt',
    'ia_active', 'ih_active', 'pa_active', 'hm_active',
    'fema_response_breadth', 'fema_program_count',
    'had_fema_flood', 'had_fema_hurricane', 'had_fema_severe_storm',
    'had_fema_fire', 'had_fema_tornado', 'had_fema_earthquake',
    
    # Rolling disaster exposure
    'log_damage_roll3m', 'log_damage_roll6m', 'log_damage_roll12m',
    'storms_roll3m', 'storms_roll6m', 'storms_roll12m',
    'fema_days_roll3m', 'fema_days_roll6m', 'fema_days_roll12m',
    'fema_decl_roll3m', 'fema_decl_roll6m', 'fema_decl_roll12m',
    
    # Pre-disaster price context (controls)
    'pct_pre3m', 'pct_pre6m', 'pct_pre12m',
    'price_ratio_national', 'log_price_level',
    
    # Economic control
    'unemployment_rate_monthly',
    
    # Weather context
    'tmax_f', 'tmin_f', 'precip_in', 'temp_range',
    'is_extreme_heat', 'is_extreme_cold', 'heavy_precip',
    
    # Seasonality
    'month', 'quarter',
]

# --- FILTER: Only disaster-affected CBSA-months ---
# A row qualifies if it has any storm activity OR any FEMA involvement
disaster_mask = (
    (df['n_storm_events'] > 0) | 
    (df['n_fema_declarations'] > 0) |
    (df['damage_property_sum'] > 0)
)

df_disaster = df[disaster_mask & df[target_col].notna()].copy()
print(f"\nDisaster-affected rows: {len(df_disaster):,} out of {len(df):,} ({100*len(df_disaster)/len(df):.1f}%)")

# Also keep ALL rows for comparison model
df_all = df[df[target_col].notna()].copy()
print(f"All rows with target: {len(df_all):,}")

# Handle missing features
for col in feature_cols:
    if col not in df_disaster.columns:
        print(f"  WARNING: {col} not found, creating as 0")
        df_disaster[col] = 0
        df_all[col] = 0

# Fill NaN in features
for dataset in [df_disaster, df_all]:
    for col in feature_cols:
        if dataset[col].isnull().any():
            dataset[col] = dataset.groupby('CBSA_CODE')[col].transform(
                lambda x: x.fillna(x.median())
            )
            dataset[col] = dataset[col].fillna(0)

print(f"\nFeatures: {len(feature_cols)}")

# ============================================================================
# 5. TEMPORAL TRAIN/TEST SPLIT
# ============================================================================

split_date = pd.to_datetime('2022-01-01')

# Disaster-only dataset
train_d = df_disaster[df_disaster['YEAR_MONTH'] < split_date]
test_d = df_disaster[df_disaster['YEAR_MONTH'] >= split_date]

X_train_d = train_d[feature_cols].values
X_test_d = test_d[feature_cols].values
y_train_d = train_d[target_col].values
y_test_d = test_d[target_col].values

# All-rows dataset (for comparison)
train_a = df_all[df_all['YEAR_MONTH'] < split_date]
test_a = df_all[df_all['YEAR_MONTH'] >= split_date]

X_train_a = train_a[feature_cols].values
X_test_a = test_a[feature_cols].values
y_train_a = train_a[target_col].values
y_test_a = test_a[target_col].values

print(f"\n{'='*60}")
print("TEMPORAL SPLIT")
print(f"{'='*60}")
print(f"\nDisaster-only dataset:")
print(f"  Train: {len(train_d):,} rows ({train_d['YEAR_MONTH'].min().date()} to {train_d['YEAR_MONTH'].max().date()})")
print(f"  Test:  {len(test_d):,} rows ({test_d['YEAR_MONTH'].min().date()} to {test_d['YEAR_MONTH'].max().date()})")
print(f"\nAll-rows dataset:")
print(f"  Train: {len(train_a):,} rows")
print(f"  Test:  {len(test_a):,} rows")

# ============================================================================
# 6. TRAIN MODELS
# ============================================================================

print(f"\n{'='*60}")
print("TRAINING MODELS")
print(f"{'='*60}")

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Baseline: predicting 0 (= national mean, i.e. no local effect)
    rmse_baseline = np.sqrt(mean_squared_error(y_true, np.zeros_like(y_true)))
    skill = 1 - (rmse / rmse_baseline)  # Forecast skill score
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 
            'RMSE_baseline': rmse_baseline, 'Skill': skill}

results_disaster = []
results_all = []
models_disaster = {}

# --- Baseline: predict 0 (national mean = no local effect) ---
print("\n[0] Baseline: predict national mean (abnormal return = 0)")
res = evaluate("Baseline (predict 0)", y_test_d, np.zeros_like(y_test_d))
results_disaster.append(res)
print(f"  RMSE: {res['RMSE']:.4f}, R²: {res['R²']:.4f}")

# --- Ridge Regression ---
print("\n[1] Ridge Regression...")
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_d)
X_te_sc = scaler.transform(X_test_d)

ridge = Ridge(alpha=10.0)
ridge.fit(X_tr_sc, y_train_d)
y_pred_ridge = ridge.predict(X_te_sc)
models_disaster['Ridge'] = ridge
res = evaluate("Ridge Regression", y_test_d, y_pred_ridge)
results_disaster.append(res)
print(f"  RMSE: {res['RMSE']:.4f}, MAE: {res['MAE']:.4f}, R²: {res['R²']:.4f}, Skill: {res['Skill']:.4f}")

# --- Random Forest ---
print("\n[2] Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_split=20,
    min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train_d, y_train_d)
y_pred_rf = rf.predict(X_test_d)
models_disaster['Random Forest'] = rf
res = evaluate("Random Forest", y_test_d, y_pred_rf)
results_disaster.append(res)
print(f"  RMSE: {res['RMSE']:.4f}, MAE: {res['MAE']:.4f}, R²: {res['R²']:.4f}, Skill: {res['Skill']:.4f}")

# --- XGBoost ---
print("\n[3] XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
    min_child_weight=20, random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train_d, y_train_d, 
              eval_set=[(X_test_d, y_test_d)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test_d)
models_disaster['XGBoost'] = xgb_model
res = evaluate("XGBoost", y_test_d, y_pred_xgb)
results_disaster.append(res)
print(f"  RMSE: {res['RMSE']:.4f}, MAE: {res['MAE']:.4f}, R²: {res['R²']:.4f}, Skill: {res['Skill']:.4f}")

# --- Gradient Boosting (sklearn, different algo) ---
print("\n[4] Gradient Boosting...")
gb = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=20, random_state=42
)
gb.fit(X_train_d, y_train_d)
y_pred_gb = gb.predict(X_test_d)
models_disaster['Gradient Boosting'] = gb
res = evaluate("Gradient Boosting", y_test_d, y_pred_gb)
results_disaster.append(res)
print(f"  RMSE: {res['RMSE']:.4f}, MAE: {res['MAE']:.4f}, R²: {res['R²']:.4f}, Skill: {res['Skill']:.4f}")

# ============================================================================
# 7. COMPARISON: Same models on ALL rows (not just disaster)
# ============================================================================

print(f"\n{'='*60}")
print("COMPARISON: Models on ALL rows (not disaster-filtered)")
print(f"{'='*60}")

res_base_all = evaluate("Baseline (all)", y_test_a, np.zeros_like(y_test_a))
results_all.append(res_base_all)

scaler_a = StandardScaler()
X_tr_a_sc = scaler_a.fit_transform(X_train_a)
X_te_a_sc = scaler_a.transform(X_test_a)

xgb_all = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
    min_child_weight=20, random_state=42, n_jobs=-1, verbosity=0
)
xgb_all.fit(X_train_a, y_train_a, eval_set=[(X_test_a, y_test_a)], verbose=False)
y_pred_xgb_all = xgb_all.predict(X_test_a)
res = evaluate("XGBoost (all rows)", y_test_a, y_pred_xgb_all)
results_all.append(res)
print(f"XGBoost all rows: RMSE={res['RMSE']:.4f}, R²={res['R²']:.4f}, Skill={res['Skill']:.4f}")

# ============================================================================
# 8. RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print(f"{'='*80}")

results_df = pd.DataFrame(results_disaster)
print("\n--- Disaster-Affected CBSA-Months Only ---")
print(results_df[['Model','RMSE','MAE','R²','Skill']].to_string(index=False))

best_idx = results_df.loc[results_df.index > 0, 'R²'].idxmax()  # exclude baseline
best = results_df.loc[best_idx]
print(f"\nBest model: {best['Model']} (R²={best['R²']:.4f})")

print("\n--- All Rows (for comparison) ---")
results_all_df = pd.DataFrame(results_all)
print(results_all_df[['Model','RMSE','MAE','R²','Skill']].to_string(index=False))

# ============================================================================
# 9. FEATURE IMPORTANCE (XGBoost)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE IMPORTANCE (XGBoost, Disaster-Only)")
print(f"{'='*80}")

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 features:")
print(importance_df.head(20).to_string(index=False))

# RF importance for comparison
rf_imp = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print("\nRandom Forest Top 20:")
print(rf_imp.head(20).to_string(index=False))

# ============================================================================
# 10. DISASTER-TYPE ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("PRICE IMPACT BY DISASTER TYPE (Test Set)")
print(f"{'='*80}")

disaster_types = ['had_tornado','had_hurricane','had_flood','had_drought',
                  'had_heat','had_winter_storm','had_wildfire']

for dtype in disaster_types:
    mask = test_d[dtype].values == 1
    if mask.sum() > 0:
        actual = y_test_d[mask]
        predicted = y_pred_xgb[mask]
        print(f"\n{dtype} (n={mask.sum()}):")
        print(f"  Actual mean abnormal return:    {actual.mean():+.4f}%")
        print(f"  Predicted mean abnormal return:  {predicted.mean():+.4f}%")
        print(f"  Actual median:                   {np.median(actual):+.4f}%")
        print(f"  Model RMSE on this subset:       {np.sqrt(mean_squared_error(actual, predicted)):.4f}")

# ============================================================================
# 11. EXTREME EVENT ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("EXTREME EVENT ANALYSIS (Test Set, Top 5% Damage)")
print(f"{'='*80}")

damage_thresh = test_d['log_damage_property'].quantile(0.95)
extreme_mask = test_d['log_damage_property'].values >= damage_thresh

if extreme_mask.sum() > 0:
    actual_ext = y_test_d[extreme_mask]
    pred_ext = y_pred_xgb[extreme_mask]
    print(f"Extreme events (n={extreme_mask.sum()}):")
    print(f"  Actual mean abnormal return:    {actual_ext.mean():+.4f}%")
    print(f"  Predicted mean abnormal return:  {pred_ext.mean():+.4f}%")
    print(f"  R² on extreme subset:           {r2_score(actual_ext, pred_ext):.4f}")
    print(f"  RMSE on extreme subset:         {np.sqrt(mean_squared_error(actual_ext, pred_ext)):.4f}")

# High-casualty events
cas_mask = test_d['has_casualties'].values == 1
if cas_mask.sum() > 0:
    print(f"\nCasualty events (n={cas_mask.sum()}):")
    print(f"  Actual mean abnormal return:    {y_test_d[cas_mask].mean():+.4f}%")
    print(f"  Predicted mean:                 {y_pred_xgb[cas_mask].mean():+.4f}%")

# ============================================================================
# 12. SAVE OUTPUTS
# ============================================================================

output_dir = "Model Analysis/Disaster_Impact_Model"
os.makedirs(output_dir, exist_ok=True)

# Model comparison
results_df.to_csv(f"{output_dir}/01_Model_Comparison.csv", index=False)
results_all_df.to_csv(f"{output_dir}/02_Model_Comparison_AllRows.csv", index=False)

# Feature importance
importance_df.to_csv(f"{output_dir}/03_XGB_Feature_Importance.csv", index=False)
rf_imp.to_csv(f"{output_dir}/04_RF_Feature_Importance.csv", index=False)

# Predictions
pred_df = pd.DataFrame({
    'YEAR_MONTH': test_d['YEAR_MONTH'].values,
    'CBSA_CODE': test_d['CBSA_CODE'].values,
    'CBSA_TITLE': test_d['CBSA_TITLE'].values,
    'Actual_Abnormal_Return': y_test_d,
    'Predicted_Ridge': y_pred_ridge,
    'Predicted_RF': y_pred_rf,
    'Predicted_XGB': y_pred_xgb,
    'Predicted_GB': y_pred_gb,
    'damage_property_sum': test_d['damage_property_sum'].values,
    'n_fema_declarations': test_d['n_fema_declarations'].values,
    'had_hurricane': test_d['had_hurricane'].values,
    'had_flood': test_d['had_flood'].values,
    'had_wildfire': test_d['had_wildfire'].values,
})
pred_df.to_csv(f"{output_dir}/05_Test_Predictions.csv", index=False)

# Disaster type analysis
dtype_results = []
for dtype in disaster_types:
    mask = test_d[dtype].values == 1
    if mask.sum() > 0:
        dtype_results.append({
            'Disaster_Type': dtype,
            'N_Events': mask.sum(),
            'Mean_Actual_Abnormal': y_test_d[mask].mean(),
            'Mean_Predicted_Abnormal': y_pred_xgb[mask].mean(),
            'Median_Actual': np.median(y_test_d[mask]),
            'RMSE': np.sqrt(mean_squared_error(y_test_d[mask], y_pred_xgb[mask])),
        })
pd.DataFrame(dtype_results).to_csv(f"{output_dir}/06_Disaster_Type_Analysis.csv", index=False)

print(f"\nAll outputs saved to: {output_dir}/")

# ============================================================================
# 13. SUMMARY REPORT
# ============================================================================

report = f"""
{'='*80}
DISASTER IMPACT MODEL — SUMMARY REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESEARCH QUESTION:
  Do natural disasters cause abnormal price movements in affected
  US housing markets? Which disaster characteristics predict the
  magnitude of price impact?

METHODOLOGY:
  Target Variable: Abnormal Return (6-month)
    = Local CBSA price change (%) — National mean price change (%)
    This controls for the macro housing cycle (rates, national demand)
    and isolates the LOCAL deviation caused by the disaster.
    
  Sample: Only disaster-affected CBSA-months
    (at least 1 storm event, FEMA declaration, or property damage > 0)
    This focuses the model on the relevant population.
    
  Features ({len(feature_cols)} total):
    - Disaster severity: damage, casualties, storm count
    - Disaster type: hurricane, flood, tornado, wildfire, etc.
    - FEMA response: declarations, active days, program types
    - Rolling exposure: 3/6/12-month cumulative disaster history
    - Controls: pre-disaster price trend, price level, unemployment
    - Weather: temperature, precipitation, extremes
    - Seasonality: month, quarter

DATA:
  Total rows: {len(df):,}
  Disaster-affected rows: {len(df_disaster):,} ({100*len(df_disaster)/len(df):.1f}%)
  Training: {len(train_d):,} rows (before 2022-01-01)
  Testing: {len(test_d):,} rows (2022-01-01 onwards)

RESULTS (Disaster-Affected Rows):
{results_df[['Model','RMSE','MAE','R²','Skill']].to_string(index=False)}

Best model: {best['Model']}
  R² = {best['R²']:.4f}
  Forecast Skill = {best['Skill']:.4f} (improvement over baseline)

COMPARISON (All Rows):
{results_all_df[['Model','RMSE','R²','Skill']].to_string(index=False)}

TOP 10 FEATURES (XGBoost):
{importance_df.head(10).to_string(index=False)}

OUTPUT FILES:
  01_Model_Comparison.csv
  02_Model_Comparison_AllRows.csv
  03_XGB_Feature_Importance.csv
  04_RF_Feature_Importance.csv
  05_Test_Predictions.csv
  06_Disaster_Type_Analysis.csv
{'='*80}
"""

with open(f"{output_dir}/00_Summary_Report.txt", 'w') as f:
    f.write(report)
    
print(report)
print("DONE.")
