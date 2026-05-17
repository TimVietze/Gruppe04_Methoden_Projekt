"""
Damage Prediction Model (6-month horizon)

Predicts whether property damage will occur in the next 6 months (classification)
and how severe it will be (regression), using only climate, disaster, and FEMA data
as features. No housing price columns are used. This establishes whether climate/disaster
data contains predictive signal independent of housing prices.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Modeling_Table_Damage.csv"
OUT_DIR = BASE_DIR

EXCLUDED_PREFIXES = ["zhvi_"]
EXCLUDED_COLS = {
    "CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH",
    "price_change_next_6m", "zhvi_avg_next_6m",
    "had_damage_next_6m", "damage_sum_next_6m", "log_damage_next_6m",
}

FOLDS = [
    ("2014-12", "2015-01", "2015-12"),
    ("2015-12", "2016-01", "2016-12"),
    ("2016-12", "2017-01", "2017-12"),
    ("2017-12", "2018-01", "2018-12"),
    ("2018-12", "2019-01", "2019-12"),
    ("2019-12", "2020-01", "2020-12"),
    ("2020-12", "2021-01", "2021-12"),
    ("2021-12", "2022-01", "2022-12"),
    ("2022-12", "2023-01", "2023-12"),
    ("2023-12", "2024-01", "2024-12"),
    ("2024-12", "2025-01", "2025-08"),
]


def separator(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# ─────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────
separator("LOADING DATA")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from {DATA_PATH.name}")

df = df.sort_values(["CBSA_CODE", "YEAR_MONTH"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
#  PART 1 — TARGET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────
separator("PART 1 — TARGET CONSTRUCTION")

def build_targets(group):
    g = group.sort_values("YEAR_MONTH").copy()
    dmg_binary = (g["damage_property_sum"] > 0).astype(int)

    # Shift by -1: position i gets the value from i+1 (looking forward)
    shifted_binary = dmg_binary.shift(-1)
    # Rolling sum over 6 periods on the shifted series: covers t+1..t+6
    fwd_any = shifted_binary.iloc[::-1].rolling(6, min_periods=1).sum().iloc[::-1]
    g["had_damage_next_6m"] = (fwd_any > 0).astype(float)

    shifted_dmg = g["damage_property_sum"].shift(-1)
    fwd_sum = shifted_dmg.iloc[::-1].rolling(6, min_periods=6).sum().iloc[::-1]
    g["damage_sum_next_6m"] = fwd_sum

    # NaN out incomplete windows (last 6 rows per CBSA)
    last_6_idx = g.index[-6:]
    g.loc[last_6_idx, "had_damage_next_6m"] = np.nan
    g.loc[last_6_idx, "damage_sum_next_6m"] = np.nan

    g["log_damage_next_6m"] = np.log1p(g["damage_sum_next_6m"])
    return g

df = df.groupby("CBSA_CODE", group_keys=False).apply(build_targets)
df = df.reset_index(drop=True)

valid_mask = df["had_damage_next_6m"].notna()
n_valid = valid_mask.sum()
n_nan = (~valid_mask).sum()
pos_rate = df.loc[valid_mask, "had_damage_next_6m"].mean()

print(f"Rows with valid targets: {n_valid:,}")
print(f"Rows with NaN targets (prediction set): {n_nan:,}")
print(f"Class balance (had_damage_next_6m=1): {pos_rate:.1%}")

dmg_pos = df.loc[valid_mask & (df["damage_sum_next_6m"] > 0), "damage_sum_next_6m"]
print(f"\ndamage_sum_next_6m where > 0 (n={len(dmg_pos):,}):")
print(f"  mean=${dmg_pos.mean():,.0f}  median=${dmg_pos.median():,.0f}")
print(f"  std=${dmg_pos.std():,.0f}  min=${dmg_pos.min():,.0f}  max=${dmg_pos.max():,.0f}")

# Verification: print 12 consecutive rows for 2 CBSAs
print("\n--- TARGET VERIFICATION ---")
for cbsa_code in [31080, 12060]:
    sub = df[df["CBSA_CODE"] == cbsa_code].sort_values("YEAR_MONTH").reset_index(drop=True)
    title = sub["CBSA_TITLE"].iloc[0]
    start_idx = 60  # pick a row well into the series
    chunk = sub.iloc[start_idx:start_idx + 12]
    print(f"\nCBSA {cbsa_code} ({title}), rows {start_idx}-{start_idx + 11}:")
    print(f"{'YEAR_MONTH':>12}  {'dmg_prop':>14}  {'had_dmg_6m':>10}  {'dmg_sum_6m':>14}")
    for _, r in chunk.iterrows():
        dmg = f"${r['damage_property_sum']:,.0f}"
        had = f"{r['had_damage_next_6m']:.0f}" if pd.notna(r["had_damage_next_6m"]) else "NaN"
        dsum = f"${r['damage_sum_next_6m']:,.0f}" if pd.notna(r["damage_sum_next_6m"]) else "NaN"
        print(f"{r['YEAR_MONTH']:>12}  {dmg:>14}  {had:>10}  {dsum:>14}")

    # Manual check for first row in chunk
    r0_idx = start_idx
    fwd_months = sub.iloc[r0_idx + 1 : r0_idx + 7]
    any_dmg = (fwd_months["damage_property_sum"] > 0).any()
    sum_dmg = fwd_months["damage_property_sum"].sum()
    row0 = sub.iloc[r0_idx]
    print(f"\n  CHECK row {r0_idx} ({row0['YEAR_MONTH']}):")
    print(f"    damage_property_sum in t+1..t+6: {fwd_months['damage_property_sum'].tolist()}")
    print(f"    Any > 0? {any_dmg}  → had_damage_next_6m should be {1 if any_dmg else 0}, actual = {row0['had_damage_next_6m']:.0f}")
    print(f"    Sum = ${sum_dmg:,.0f}  → damage_sum_next_6m should be ${sum_dmg:,.0f}, actual = ${row0['damage_sum_next_6m']:,.0f}")


# ─────────────────────────────────────────────────────────────────────
#  PART 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
separator("PART 2 — FEATURE ENGINEERING")

def compute_rolling_features(group):
    g = group.sort_values("YEAR_MONTH").copy()

    # Weather rolling (12m)
    g["avg_tmax_last_12m"] = g["tmax_f"].rolling(12, min_periods=1).mean()
    g["avg_tmin_last_12m"] = g["tmin_f"].rolling(12, min_periods=1).mean()
    g["total_precip_last_12m"] = g["precip_in"].rolling(12, min_periods=1).sum()

    g["extreme_heat_months_last_12m"] = (g["tmax_f"] > 95).astype(int).rolling(12, min_periods=1).sum()

    precip_90th = g["precip_in"].quantile(0.9)
    g["extreme_precip_months_last_12m"] = (g["precip_in"] > precip_90th).astype(int).rolling(12, min_periods=1).sum()

    # Hazard rolling (12m)
    for haz in ["flood", "wildfire", "hurricane", "heat", "tornado", "winter_storm", "drought"]:
        g[f"{haz}_months_last_12m"] = g[f"had_{haz}"].rolling(12, min_periods=1).sum()

    # FEMA rolling (3m, 6m, 12m)
    for col in ["n_fema_declarations", "fema_active_days"]:
        for w in [3, 6, 12]:
            g[f"{col}_last_{w}m"] = g[col].rolling(w, min_periods=1).sum()

    # Storm and damage rolling
    for col in ["n_storm_events", "damage_property_sum"]:
        for w in [3, 6, 12]:
            g[f"{col}_last_{w}m"] = g[col].rolling(w, min_periods=1).sum()

    g["deaths_total_last_12m"] = g["deaths_total"].rolling(12, min_periods=1).sum()
    g["injuries_total_last_12m"] = g["injuries_total"].rolling(12, min_periods=1).sum()

    return g

df = df.groupby("CBSA_CODE", group_keys=False).apply(compute_rolling_features)
df = df.reset_index(drop=True)

# Seasonal
df["_month"] = pd.to_datetime(df["YEAR_MONTH"]).dt.month
df["month_sin"] = np.sin(2 * np.pi * df["_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["_month"] / 12)
df.drop(columns=["_month"], inplace=True)

# Verify rolling direction
print("--- ROLLING DIRECTION VERIFICATION ---")
cbsa_la = df[df["CBSA_CODE"] == 31080].sort_values("YEAR_MONTH").reset_index(drop=True)
row_jun2015 = cbsa_la[cbsa_la["YEAR_MONTH"] == "2015-06"]
if len(row_jun2015) > 0:
    r = row_jun2015.iloc[0]
    window_start = "2014-07"
    window_end = "2015-06"
    window_rows = cbsa_la[(cbsa_la["YEAR_MONTH"] >= window_start) & (cbsa_la["YEAR_MONTH"] <= window_end)]
    manual_sum = window_rows["damage_property_sum"].sum()
    print(f"CBSA 31080, June 2015:")
    print(f"  damage_property_sum_last_12m = {r['damage_property_sum_last_12m']:,.0f}")
    print(f"  Manual sum (Jul 2014 – Jun 2015) = {manual_sum:,.0f}")
    print(f"  Match: {abs(r['damage_property_sum_last_12m'] - manual_sum) < 1}")

    after_jun = cbsa_la[cbsa_la["YEAR_MONTH"] > "2015-06"].head(3)
    print(f"  Months AFTER Jun 2015: {after_jun['YEAR_MONTH'].tolist()} — damage: {after_jun['damage_property_sum'].tolist()}")
    print(f"  These are NOT included in the rolling sum (confirmed backward-looking).")

# Build feature list
CURRENT_RAW = [
    "tmax_f", "tmin_f", "precip_in",
    "n_storm_events", "damage_property_sum", "damage_crops_sum",
    "deaths_total", "injuries_total",
    "had_tornado", "had_hurricane", "had_flood", "had_drought",
    "had_heat", "had_winter_storm", "had_wildfire",
    "n_fema_declarations", "fema_active_days",
    "had_major_disaster", "had_emergency", "had_fire_mgmt",
    "ia_active", "ih_active", "pa_active", "hm_active",
    "had_fema_flood", "had_fema_hurricane", "had_fema_severe_storm",
    "had_fema_fire", "had_fema_tornado", "had_fema_earthquake",
    "had_fema_biological",
    "unemployment_rate_monthly",
]

COMPUTED_ROLLING = [
    "avg_tmax_last_12m", "avg_tmin_last_12m", "total_precip_last_12m",
    "extreme_heat_months_last_12m", "extreme_precip_months_last_12m",
    "flood_months_last_12m", "wildfire_months_last_12m",
    "hurricane_months_last_12m", "heat_months_last_12m",
    "tornado_months_last_12m", "winter_storm_months_last_12m",
    "drought_months_last_12m",
    "n_fema_declarations_last_3m", "n_fema_declarations_last_6m", "n_fema_declarations_last_12m",
    "fema_active_days_last_3m", "fema_active_days_last_6m", "fema_active_days_last_12m",
    "n_storm_events_last_3m", "n_storm_events_last_6m", "n_storm_events_last_12m",
    "damage_property_sum_last_3m", "damage_property_sum_last_6m", "damage_property_sum_last_12m",
    "deaths_total_last_12m", "injuries_total_last_12m",
]

SEASONAL = ["month_sin", "month_cos"]

FEATURES_STAGE1 = CURRENT_RAW + COMPUTED_ROLLING + SEASONAL

print(f"\n--- FEATURE LIST (Stage 1): {len(FEATURES_STAGE1)} features ---")
print("Current raw:", len(CURRENT_RAW))
print("Computed rolling:", len(COMPUTED_ROLLING))
print("Seasonal:", len(SEASONAL))

# Exclusion check
violations = []
for f in FEATURES_STAGE1:
    if "next_6m" in f:
        violations.append(f)
    if f.startswith("zhvi_"):
        violations.append(f)
    if f in {"CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH", "price_change_next_6m"}:
        violations.append(f)

if violations:
    print(f"\n*** EXCLUSION VIOLATION: {violations} ***")
    raise ValueError("Excluded columns found in feature list!")
else:
    print("\nExclusion check PASSED: no zhvi, no next_6m, no identifiers, no price targets.")


# ─────────────────────────────────────────────────────────────────────
#  PART 3 — STAGE 1: DAMAGE OCCURRENCE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────
separator("PART 3 — STAGE 1: DAMAGE OCCURRENCE CLASSIFIER")

df_valid = df[df["had_damage_next_6m"].notna()].copy()
null_frac = df_valid[FEATURES_STAGE1].isnull().mean(axis=1)
df_valid = df_valid[null_frac <= 0.5].copy()
print(f"Rows after dropping >50% null features: {len(df_valid):,}")

# Determine last valid target month
last_valid_ym = df_valid["YEAR_MONTH"].max()
print(f"Last month with valid targets: {last_valid_ym}")

# Adjust final fold if needed
adjusted_folds = []
for train_end, test_start, test_end in FOLDS:
    if test_start > last_valid_ym:
        break
    actual_test_end = min(test_end, last_valid_ym)
    adjusted_folds.append((train_end, test_start, actual_test_end))
print(f"Number of folds: {len(adjusted_folds)}")


def optimal_f1_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_f1 = 0
    best_t = 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t


def eval_classifier(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "AP": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "Precision@0.5": precision_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0),
        "Recall@0.5": recall_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0),
        "F1@0.5": f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0),
        "Threshold_opt": threshold,
        "Precision@opt": precision_score(y_true, y_pred, zero_division=0),
        "Recall@opt": recall_score(y_true, y_pred, zero_division=0),
        "F1@opt": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy@opt": accuracy_score(y_true, y_pred),
    }


# Store out-of-fold predictions
df_valid["oof_prob_naive"] = np.nan
df_valid["oof_prob_lr"] = np.nan
df_valid["oof_prob_rf"] = np.nan
df_valid["oof_prob_hgb"] = np.nan

clf_results = []

print("\n--- A. CLASS BALANCE PER FOLD ---")
print(f"{'Fold':>6}  {'Train end':>10}  {'Test':>22}  {'Train N':>8}  {'Test N':>7}  {'Train %+':>8}  {'Test %+':>7}")

for fold_i, (train_end, test_start, test_end) in enumerate(adjusted_folds, 1):
    train_mask = df_valid["YEAR_MONTH"] <= train_end
    test_mask = (df_valid["YEAR_MONTH"] >= test_start) & (df_valid["YEAR_MONTH"] <= test_end)

    train_df = df_valid[train_mask]
    test_df = df_valid[test_mask]

    if len(test_df) == 0:
        continue

    X_train = train_df[FEATURES_STAGE1].values
    y_train = train_df["had_damage_next_6m"].values.astype(int)
    X_test = test_df[FEATURES_STAGE1].values
    y_test = test_df["had_damage_next_6m"].values.astype(int)

    train_pos = y_train.mean()
    test_pos = y_test.mean()
    print(f"{fold_i:>6}  {train_end:>10}  {test_start}–{test_end:>10}  {len(y_train):>8,}  {len(y_test):>7,}  {train_pos:>7.1%}  {test_pos:>6.1%}")

    test_idx = test_df.index

    # 1. Naive baseline
    naive_prob = np.full(len(y_test), train_pos)
    df_valid.loc[test_idx, "oof_prob_naive"] = naive_prob
    opt_t_naive = optimal_f1_threshold(y_train, np.full(len(y_train), train_pos))
    metrics_naive = eval_classifier(y_test, naive_prob, threshold=opt_t_naive)
    metrics_naive["model"] = "Naive"
    metrics_naive["fold"] = fold_i
    clf_results.append(metrics_naive)

    # Shared preprocessing
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    # 2. Logistic Regression with inner CV for C
    best_c, best_auc_inner = 1.0, -1
    for c_val in [0.01, 0.1, 1.0, 10.0]:
        n_inner = min(3, max(1, int(len(y_train) / 5000)))
        inner_size = len(y_train) // (n_inner + 1)
        inner_aucs = []
        for k in range(n_inner):
            inner_train_end = inner_size * (k + 1)
            inner_val_end = min(inner_train_end + inner_size, len(y_train))
            if inner_val_end <= inner_train_end:
                continue
            Xti = X_train_sc[:inner_train_end]
            yti = y_train[:inner_train_end]
            Xvi = X_train_sc[inner_train_end:inner_val_end]
            yvi = y_train[inner_train_end:inner_val_end]
            if len(np.unique(yvi)) < 2 or len(np.unique(yti)) < 2:
                continue
            lr_inner = LogisticRegression(C=c_val, class_weight="balanced", max_iter=1000, solver="lbfgs")
            lr_inner.fit(Xti, yti)
            inner_aucs.append(roc_auc_score(yvi, lr_inner.predict_proba(Xvi)[:, 1]))
        if inner_aucs and np.mean(inner_aucs) > best_auc_inner:
            best_auc_inner = np.mean(inner_aucs)
            best_c = c_val

    lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000, solver="lbfgs")
    lr.fit(X_train_sc, y_train)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]
    df_valid.loc[test_idx, "oof_prob_lr"] = lr_prob
    opt_t_lr = optimal_f1_threshold(y_train, lr.predict_proba(X_train_sc)[:, 1])
    metrics_lr = eval_classifier(y_test, lr_prob, threshold=opt_t_lr)
    metrics_lr["model"] = "LogisticRegression"
    metrics_lr["fold"] = fold_i
    clf_results.append(metrics_lr)

    # 3. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_imp, y_train)
    rf_prob = rf.predict_proba(X_test_imp)[:, 1]
    df_valid.loc[test_idx, "oof_prob_rf"] = rf_prob
    opt_t_rf = optimal_f1_threshold(y_train, rf.predict_proba(X_train_imp)[:, 1])
    metrics_rf = eval_classifier(y_test, rf_prob, threshold=opt_t_rf)
    metrics_rf["model"] = "RandomForest"
    metrics_rf["fold"] = fold_i
    clf_results.append(metrics_rf)

    # 4. HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05, random_state=42,
    )
    hgb.fit(X_train_imp, y_train)
    hgb_prob = hgb.predict_proba(X_test_imp)[:, 1]
    df_valid.loc[test_idx, "oof_prob_hgb"] = hgb_prob
    opt_t_hgb = optimal_f1_threshold(y_train, hgb.predict_proba(X_train_imp)[:, 1])
    metrics_hgb = eval_classifier(y_test, hgb_prob, threshold=opt_t_hgb)
    metrics_hgb["model"] = "HistGradientBoosting"
    metrics_hgb["fold"] = fold_i
    clf_results.append(metrics_hgb)

clf_df = pd.DataFrame(clf_results)

# B. Full results
print("\n--- B. FULL RESULTS: STAGE 1 ---")
metric_cols = ["AUC-ROC", "AP", "F1@0.5", "Threshold_opt", "F1@opt", "Precision@opt", "Recall@opt", "Accuracy@opt"]
print(clf_df[["model", "fold"] + metric_cols].to_string(index=False, float_format="%.4f"))

# C. Summary
print("\n--- C. SUMMARY: STAGE 1 ---")
summary1 = clf_df.groupby("model").agg(
    mean_AUC_ROC=("AUC-ROC", "mean"),
    std_AUC_ROC=("AUC-ROC", "std"),
    mean_AP=("AP", "mean"),
    mean_F1_opt=("F1@opt", "mean"),
).reset_index().sort_values("mean_AUC_ROC", ascending=False)
print(summary1.to_string(index=False, float_format="%.4f"))

# D. Best classifier
best_clf_name = summary1.iloc[0]["model"]
best_clf_auc = summary1.iloc[0]["mean_AUC_ROC"]
naive_auc = summary1[summary1["model"] == "Naive"]["mean_AUC_ROC"].values[0]
delta_auc = best_clf_auc - naive_auc

print(f"\n--- D. BEST CLASSIFIER: {best_clf_name} ---")
print(f"Mean AUC-ROC: {best_clf_auc:.4f}")
print(f"Naive AUC-ROC: {naive_auc:.4f}")
print(f"Delta: {delta_auc:.4f}")

# E. Top 20 features from best tree model
print("\n--- E. TOP 20 FEATURE IMPORTANCES (best tree model) ---")
X_all_valid = df_valid[FEATURES_STAGE1].values
y_all_valid = df_valid["had_damage_next_6m"].values.astype(int)

imp_all = SimpleImputer(strategy="median")
X_all_imp = imp_all.fit_transform(X_all_valid)

IMP_SAMPLE = 30_000
rng = np.random.RandomState(42)
if len(X_all_imp) > IMP_SAMPLE:
    idx_imp = rng.choice(len(X_all_imp), IMP_SAMPLE, replace=False)
    X_imp_sub, y_imp_sub = X_all_imp[idx_imp], y_all_valid[idx_imp]
else:
    X_imp_sub, y_imp_sub = X_all_imp, y_all_valid

if "RandomForest" in best_clf_name:
    imp_model = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    imp_model.fit(X_imp_sub, y_imp_sub)
    importances = imp_model.feature_importances_
else:
    imp_model = HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    imp_model.fit(X_imp_sub, y_imp_sub)
    perm_result = permutation_importance(imp_model, X_imp_sub, y_imp_sub, n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc")
    importances = perm_result.importances_mean

feat_imp_clf = pd.DataFrame({"feature": FEATURES_STAGE1, "importance": importances})
feat_imp_clf = feat_imp_clf.sort_values("importance", ascending=False).reset_index(drop=True)
print(feat_imp_clf.head(20).to_string(index=False, float_format="%.4f"))

# F. Decision Gate
print("\n--- F. DECISION GATE ---")
print(f"Best AUC-ROC: {best_clf_auc:.4f}")
print(f"Naive AUC-ROC: {naive_auc:.4f}")
print(f"Delta: {delta_auc:.4f}")

if delta_auc >= 0.05:
    signal_strength_clf = "SIGNAL CONFIRMED"
elif delta_auc >= 0.02:
    signal_strength_clf = "MODERATE SIGNAL"
else:
    signal_strength_clf = "WARNING — WEAK SIGNAL"
print(f"Verdict: {signal_strength_clf}")


# ─────────────────────────────────────────────────────────────────────
#  PART 4 — STAGE 2: DAMAGE SEVERITY REGRESSOR
# ─────────────────────────────────────────────────────────────────────
separator("PART 4 — STAGE 2: DAMAGE SEVERITY REGRESSOR")

# Determine which OOF probability column to use based on best classifier
oof_col_map = {
    "Naive": "oof_prob_naive",
    "LogisticRegression": "oof_prob_lr",
    "RandomForest": "oof_prob_rf",
    "HistGradientBoosting": "oof_prob_hgb",
}
best_oof_col = oof_col_map.get(best_clf_name, "oof_prob_hgb")

FEATURES_STAGE2 = FEATURES_STAGE1 + ["predicted_damage_probability"]

df_valid["predicted_damage_probability"] = df_valid[best_oof_col]

df_dmg_pos = df_valid[df_valid["had_damage_next_6m"] == 1].copy()
print(f"Damage-positive rows for Stage 2: {len(df_dmg_pos):,}")
print(f"Target: log_damage_next_6m")
print(f"Features: {len(FEATURES_STAGE2)} (Stage 1 features + predicted_damage_probability)")

reg_results = []

print("\n--- A. DAMAGE-POSITIVE ROW COUNT PER FOLD ---")
print(f"{'Fold':>6}  {'Train end':>10}  {'Test':>22}  {'Train N':>8}  {'Test N':>7}")

for fold_i, (train_end, test_start, test_end) in enumerate(adjusted_folds, 1):
    train_mask = df_dmg_pos["YEAR_MONTH"] <= train_end
    test_mask = (df_dmg_pos["YEAR_MONTH"] >= test_start) & (df_dmg_pos["YEAR_MONTH"] <= test_end)

    train_df = df_dmg_pos[train_mask]
    test_df = df_dmg_pos[test_mask]

    if len(test_df) == 0:
        continue

    X_train = train_df[FEATURES_STAGE2].values
    y_train = train_df["log_damage_next_6m"].values
    X_test = test_df[FEATURES_STAGE2].values
    y_test = test_df["log_damage_next_6m"].values

    print(f"{fold_i:>6}  {train_end:>10}  {test_start}–{test_end:>10}  {len(y_train):>8,}  {len(y_test):>7,}", end="")
    if len(y_test) < 30:
        print("  ⚠ <30 rows")
    else:
        print()

    test_idx = test_df.index

    # Preprocessing
    imputer_r = SimpleImputer(strategy="median")
    X_train_imp = imputer_r.fit_transform(X_train)
    X_test_imp = imputer_r.transform(X_test)

    scaler_r = StandardScaler()
    X_train_sc = scaler_r.fit_transform(X_train_imp)
    X_test_sc = scaler_r.transform(X_test_imp)

    # Actual dollars for evaluation
    y_test_dollars = np.expm1(y_test)

    # 1. Naive baseline
    naive_pred = np.full(len(y_test), y_train.mean())
    naive_pred_dollars = np.expm1(naive_pred)
    reg_results.append({
        "model": "Naive", "fold": fold_i,
        "MAE_log": mean_absolute_error(y_test, naive_pred),
        "RMSE_log": np.sqrt(mean_squared_error(y_test, naive_pred)),
        "R2_log": r2_score(y_test, naive_pred),
        "MAE_dollars": mean_absolute_error(y_test_dollars, naive_pred_dollars),
        "MedAE_dollars": median_absolute_error(y_test_dollars, naive_pred_dollars),
    })

    # 2. Ridge
    best_alpha, best_mse_inner = 1.0, np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        n_inner = min(3, max(1, int(len(y_train) / 2000)))
        inner_size = len(y_train) // (n_inner + 1)
        inner_mses = []
        for k in range(n_inner):
            inner_train_end = inner_size * (k + 1)
            inner_val_end = min(inner_train_end + inner_size, len(y_train))
            if inner_val_end <= inner_train_end:
                continue
            Xti = X_train_sc[:inner_train_end]
            yti = y_train[:inner_train_end]
            Xvi = X_train_sc[inner_train_end:inner_val_end]
            yvi = y_train[inner_train_end:inner_val_end]
            ridge_inner = Ridge(alpha=alpha)
            ridge_inner.fit(Xti, yti)
            inner_mses.append(mean_squared_error(yvi, ridge_inner.predict(Xvi)))
        if inner_mses and np.mean(inner_mses) < best_mse_inner:
            best_mse_inner = np.mean(inner_mses)
            best_alpha = alpha

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_sc, y_train)
    ridge_pred = ridge.predict(X_test_sc)
    ridge_pred_dollars = np.expm1(ridge_pred)
    reg_results.append({
        "model": "Ridge", "fold": fold_i,
        "MAE_log": mean_absolute_error(y_test, ridge_pred),
        "RMSE_log": np.sqrt(mean_squared_error(y_test, ridge_pred)),
        "R2_log": r2_score(y_test, ridge_pred),
        "MAE_dollars": mean_absolute_error(y_test_dollars, ridge_pred_dollars),
        "MedAE_dollars": median_absolute_error(y_test_dollars, ridge_pred_dollars),
    })

    # 3. Random Forest
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_imp, y_train)
    rf_pred = rf_reg.predict(X_test_imp)
    rf_pred_dollars = np.expm1(rf_pred)
    reg_results.append({
        "model": "RandomForest", "fold": fold_i,
        "MAE_log": mean_absolute_error(y_test, rf_pred),
        "RMSE_log": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "R2_log": r2_score(y_test, rf_pred),
        "MAE_dollars": mean_absolute_error(y_test_dollars, rf_pred_dollars),
        "MedAE_dollars": median_absolute_error(y_test_dollars, rf_pred_dollars),
    })

    # 4. HistGradientBoosting
    hgb_reg = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    hgb_reg.fit(X_train_imp, y_train)
    hgb_pred = hgb_reg.predict(X_test_imp)
    hgb_pred_dollars = np.expm1(hgb_pred)
    reg_results.append({
        "model": "HistGradientBoosting", "fold": fold_i,
        "MAE_log": mean_absolute_error(y_test, hgb_pred),
        "RMSE_log": np.sqrt(mean_squared_error(y_test, hgb_pred)),
        "R2_log": r2_score(y_test, hgb_pred),
        "MAE_dollars": mean_absolute_error(y_test_dollars, hgb_pred_dollars),
        "MedAE_dollars": median_absolute_error(y_test_dollars, hgb_pred_dollars),
    })

reg_df = pd.DataFrame(reg_results)

# B. Full results
print("\n--- B. FULL RESULTS: STAGE 2 ---")
print(reg_df[["model", "fold", "MAE_log", "RMSE_log", "R2_log", "MAE_dollars", "MedAE_dollars"]].to_string(
    index=False, float_format="%.4f"
))

# C. Summary
print("\n--- C. SUMMARY: STAGE 2 ---")
summary2 = reg_df.groupby("model").agg(
    mean_MAE_log=("MAE_log", "mean"),
    std_MAE_log=("MAE_log", "std"),
    mean_RMSE_log=("RMSE_log", "mean"),
    mean_R2_log=("R2_log", "mean"),
    mean_MAE_dollars=("MAE_dollars", "mean"),
    mean_MedAE_dollars=("MedAE_dollars", "mean"),
).reset_index().sort_values("mean_MAE_log")
print(summary2.to_string(index=False, float_format="%.4f"))

# D. Best regressor
best_reg_name = summary2.iloc[0]["model"]
best_reg_mae = summary2.iloc[0]["mean_MAE_log"]
print(f"\n--- D. BEST REGRESSOR: {best_reg_name} ---")
print(f"Mean MAE (log): {best_reg_mae:.4f}")

# E. Top 20 features from best tree regressor
print("\n--- E. TOP 20 FEATURE IMPORTANCES (Stage 2) ---")
X_pos_all = df_dmg_pos[FEATURES_STAGE2].values
y_pos_all = df_dmg_pos["log_damage_next_6m"].values

imp_pos = SimpleImputer(strategy="median")
X_pos_imp = imp_pos.fit_transform(X_pos_all)

rng2 = np.random.RandomState(42)
if len(X_pos_imp) > IMP_SAMPLE:
    idx_imp2 = rng2.choice(len(X_pos_imp), IMP_SAMPLE, replace=False)
    X_pos_sub, y_pos_sub = X_pos_imp[idx_imp2], y_pos_all[idx_imp2]
else:
    X_pos_sub, y_pos_sub = X_pos_imp, y_pos_all

if "RandomForest" in best_reg_name:
    imp_reg_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    imp_reg_model.fit(X_pos_sub, y_pos_sub)
    importances_reg = imp_reg_model.feature_importances_
else:
    imp_reg_model = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    imp_reg_model.fit(X_pos_sub, y_pos_sub)
    perm_result_reg = permutation_importance(imp_reg_model, X_pos_sub, y_pos_sub, n_repeats=5, random_state=42, n_jobs=-1)
    importances_reg = perm_result_reg.importances_mean

feat_imp_reg = pd.DataFrame({"feature": FEATURES_STAGE2, "importance": importances_reg})
feat_imp_reg = feat_imp_reg.sort_values("importance", ascending=False).reset_index(drop=True)
print(feat_imp_reg.head(20).to_string(index=False, float_format="%.4f"))

# F. Sanity: top 20 actual damage vs predicted
print("\n--- F. SANITY CHECK: 20 HIGHEST ACTUAL DAMAGE CBSA-MONTHS ---")
# Get test-fold predictions for HGB regressor (retrain on all data)
# Use the last fold's model or retrain for sanity check
hgb_reg_final = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
hgb_reg_final.fit(X_pos_imp, y_pos_all)
df_dmg_pos = df_dmg_pos.copy()
df_dmg_pos["pred_log"] = hgb_reg_final.predict(X_pos_imp)
df_dmg_pos["pred_dollars"] = np.expm1(df_dmg_pos["pred_log"])
df_dmg_pos["actual_dollars"] = np.expm1(df_dmg_pos["log_damage_next_6m"])

top20 = df_dmg_pos.nlargest(20, "actual_dollars")[
    ["CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH", "actual_dollars", "pred_dollars"]
].copy()
top20["actual_rank"] = range(1, 21)
top20["pred_rank"] = top20["pred_dollars"].rank(ascending=False).astype(int)
print(top20.to_string(index=False, float_format="%.0f"))

rho, pval = spearmanr(top20["actual_dollars"], top20["pred_dollars"])
print(f"\nSpearman rank correlation (top 20): ρ = {rho:.4f}, p = {pval:.4f}")


# ─────────────────────────────────────────────────────────────────────
#  PART 5 — FINAL OUTPUT
# ─────────────────────────────────────────────────────────────────────
separator("PART 5 — FINAL OUTPUT")

# Retrain best classifier on ALL valid data
print("Retraining best classifier on all valid data...")
X_all_valid = df_valid[FEATURES_STAGE1].values
y_all_valid_cls = df_valid["had_damage_next_6m"].values.astype(int)

imp_final_cls = SimpleImputer(strategy="median")
X_all_imp_cls = imp_final_cls.fit_transform(X_all_valid)

if "HistGradient" in best_clf_name:
    final_clf = HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    final_clf.fit(X_all_imp_cls, y_all_valid_cls)
elif "RandomForest" in best_clf_name:
    final_clf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    final_clf.fit(X_all_imp_cls, y_all_valid_cls)
elif "LogisticRegression" in best_clf_name:
    scaler_final = StandardScaler()
    X_all_sc_cls = scaler_final.fit_transform(X_all_imp_cls)
    final_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs")
    final_clf.fit(X_all_sc_cls, y_all_valid_cls)
else:
    final_clf = HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    final_clf.fit(X_all_imp_cls, y_all_valid_cls)

# Retrain best regressor on ALL damage-positive valid data
print("Retraining best regressor on all damage-positive data...")
X_pos_all_s2 = df_dmg_pos[FEATURES_STAGE2].values
y_pos_all_s2 = df_dmg_pos["log_damage_next_6m"].values

imp_final_reg = SimpleImputer(strategy="median")
X_pos_imp_s2 = imp_final_reg.fit_transform(X_pos_all_s2)

if "HistGradient" in best_reg_name:
    final_reg = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    final_reg.fit(X_pos_imp_s2, y_pos_all_s2)
elif "RandomForest" in best_reg_name:
    final_reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    final_reg.fit(X_pos_imp_s2, y_pos_all_s2)
elif "Ridge" in best_reg_name:
    scaler_final_reg = StandardScaler()
    X_pos_sc_s2 = scaler_final_reg.fit_transform(X_pos_imp_s2)
    final_reg = Ridge(alpha=1.0)
    final_reg.fit(X_pos_sc_s2, y_pos_all_s2)
else:
    final_reg = HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
    final_reg.fit(X_pos_imp_s2, y_pos_all_s2)

# Score entire dataset
print("Scoring full dataset...")
X_full = df[FEATURES_STAGE1].values
X_full_imp = imp_final_cls.transform(X_full)

needs_scaling_clf = "LogisticRegression" in best_clf_name
if needs_scaling_clf:
    X_full_sc = scaler_final.transform(X_full_imp)
    full_probs = final_clf.predict_proba(X_full_sc)[:, 1]
else:
    full_probs = final_clf.predict_proba(X_full_imp)[:, 1]

# Use OOF probabilities where available, full-model elsewhere
df["predicted_damage_probability"] = full_probs
valid_oof_mask = df_valid[best_oof_col].notna()
valid_oof_idx = df_valid.loc[valid_oof_mask].index
df.loc[valid_oof_idx, "predicted_damage_probability"] = df_valid.loc[valid_oof_mask, best_oof_col].values

# Severity: score all rows, using predicted_damage_probability as extra feature
X_full_s2 = df[FEATURES_STAGE2].values
X_full_s2_imp = imp_final_reg.transform(X_full_s2)

needs_scaling_reg = "Ridge" in best_reg_name
if needs_scaling_reg:
    X_full_s2_sc = scaler_final_reg.transform(X_full_s2_imp)
    full_severity_log = final_reg.predict(X_full_s2_sc)
else:
    full_severity_log = final_reg.predict(X_full_s2_imp)

df["predicted_damage_severity_log"] = full_severity_log
df["predicted_damage_severity_dollars"] = np.expm1(full_severity_log)
df["expected_damage_6m"] = df["predicted_damage_probability"] * df["predicted_damage_severity_dollars"]
df["is_prediction_set"] = df["had_damage_next_6m"].isna()

# FILE 1: Predictions
out_cols_1 = [
    "CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH",
    "predicted_damage_probability",
    "predicted_damage_severity_log",
    "predicted_damage_severity_dollars",
    "expected_damage_6m",
    "had_damage_next_6m",
    "damage_sum_next_6m",
    "is_prediction_set",
]
df[out_cols_1].to_csv(OUT_DIR / "results_damage_predictions_6m.csv", index=False)
print(f"Saved: results_damage_predictions_6m.csv ({len(df):,} rows)")

# FILE 2: Evaluation metrics
eval_rows = []
for _, row in clf_df.iterrows():
    for metric in ["AUC-ROC", "AP", "F1@0.5", "F1@opt", "Precision@opt", "Recall@opt", "Accuracy@opt", "Threshold_opt"]:
        eval_rows.append({
            "stage": "classifier",
            "model": row["model"],
            "fold": row["fold"],
            "metric_name": metric,
            "metric_value": row[metric],
        })
for _, row in reg_df.iterrows():
    for metric in ["MAE_log", "RMSE_log", "R2_log", "MAE_dollars", "MedAE_dollars"]:
        eval_rows.append({
            "stage": "regressor",
            "model": row["model"],
            "fold": row["fold"],
            "metric_name": metric,
            "metric_value": row[metric],
        })
eval_out = pd.DataFrame(eval_rows)
eval_out.to_csv(OUT_DIR / "results_damage_evaluation_6m.csv", index=False)
print(f"Saved: results_damage_evaluation_6m.csv ({len(eval_out):,} rows)")

# FILE 3: Feature importances
feat_rows = []
for _, row in feat_imp_clf.iterrows():
    feat_rows.append({"feature": row["feature"], "importance": row["importance"], "model": best_clf_name, "stage": "classifier"})
for _, row in feat_imp_reg.iterrows():
    feat_rows.append({"feature": row["feature"], "importance": row["importance"], "model": best_reg_name, "stage": "regressor"})
feat_out = pd.DataFrame(feat_rows)
feat_out.to_csv(OUT_DIR / "results_damage_features_6m.csv", index=False)
print(f"Saved: results_damage_features_6m.csv ({len(feat_out):,} rows)")


# ─────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
separator("FINAL SUMMARY")

# 1. Classifier
print(f"1. CLASSIFIER: {best_clf_name}")
print(f"   AUC-ROC: {best_clf_auc:.4f} (naive: {naive_auc:.4f}, delta: {delta_auc:.4f})")
print(f"   Signal: {signal_strength_clf}")

# 2. Regressor
naive_mae_log = summary2[summary2["model"] == "Naive"]["mean_MAE_log"].values[0]
best_reg_r2 = summary2.iloc[0]["mean_R2_log"]
best_reg_mae_dollars = summary2.iloc[0]["mean_MAE_dollars"]
print(f"\n2. REGRESSOR: {best_reg_name}")
print(f"   MAE (log): {best_reg_mae:.4f} (naive: {naive_mae_log:.4f})")
print(f"   MAE (dollars): ${best_reg_mae_dollars:,.0f}")
print(f"   R² (log): {best_reg_r2:.4f}")

# 3. Top 15 features combined
print("\n3. TOP 15 FEATURES (combined ranking):")
feat_imp_clf["rank_clf"] = range(1, len(feat_imp_clf) + 1)
feat_imp_reg_mapped = feat_imp_reg.copy()
feat_imp_reg_mapped["rank_reg"] = range(1, len(feat_imp_reg_mapped) + 1)

combined = feat_imp_clf[["feature", "rank_clf"]].merge(
    feat_imp_reg_mapped[["feature", "rank_reg"]], on="feature", how="outer"
)
combined["rank_clf"] = combined["rank_clf"].fillna(len(FEATURES_STAGE1))
combined["rank_reg"] = combined["rank_reg"].fillna(len(FEATURES_STAGE2))
combined["avg_rank"] = (combined["rank_clf"] + combined["rank_reg"]) / 2
combined = combined.sort_values("avg_rank").head(15)
for i, (_, row) in enumerate(combined.iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:40s} clf_rank={int(row['rank_clf']):3d}  reg_rank={int(row['rank_reg']):3d}")

# 4. Top 20 CBSAs by expected_damage_6m in prediction set
print("\n4. TOP 20 CBSAs BY EXPECTED DAMAGE (prediction set):")
pred_set = df[df["is_prediction_set"]].copy()
if len(pred_set) > 0:
    top_cbsa = pred_set.sort_values("expected_damage_6m", ascending=False).head(20)
    print(f"{'CBSA':>8}  {'Title':>45}  {'Month':>8}  {'Prob':>6}  {'Exp Damage ($)':>16}")
    for _, r in top_cbsa.iterrows():
        print(f"{r['CBSA_CODE']:>8}  {str(r['CBSA_TITLE'])[:45]:>45}  {r['YEAR_MONTH']:>8}  {r['predicted_damage_probability']:>5.3f}  ${r['expected_damage_6m']:>14,.0f}")
else:
    print("   No prediction set rows found.")

# 5. Correlation
print("\n5. CORRELATION: expected_damage_6m vs damage_property_sum_last_12m")
valid_corr = df[df["damage_property_sum_last_12m"].notna() & df["expected_damage_6m"].notna()]
if len(valid_corr) > 10:
    rho_corr, p_corr = spearmanr(valid_corr["expected_damage_6m"], valid_corr["damage_property_sum_last_12m"])
    print(f"   Spearman ρ = {rho_corr:.4f}, p = {p_corr:.2e}")
else:
    print("   Insufficient data for correlation.")

# 6. Recommendation
print("\n6. RECOMMENDATION:")
if delta_auc >= 0.05 and best_reg_r2 > 0:
    overall_signal = "STRONG"
    use_in_price = "YES — include predicted_damage_probability and expected_damage_6m as features in the price model."
elif delta_auc >= 0.02:
    overall_signal = "MODERATE"
    use_in_price = "CONDITIONAL — include predicted_damage_probability in the price model. Monitor for improvement."
else:
    overall_signal = "WEAK"
    use_in_price = "NO — damage predictions do not add enough signal. Re-examine feature engineering or data granularity."

print(f"   Signal strength: {overall_signal}")
print(f"   Use in price model: {use_in_price}")

print(f"\n{'=' * 70}")
print(f"  DONE — All outputs saved to: {OUT_DIR}")
print(f"{'=' * 70}")
