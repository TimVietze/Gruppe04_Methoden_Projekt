"""
02_ml_v2_macro.py  —  Phase V2.2

scikit-learn ML prototype with FRED national macro features.

Models trained:
  Naive baseline         — predict median training target for every row
  Ridge         [A]      — housing history only (v1 baseline reproduced for comparison)
  Ridge         [B]      — housing history + FRED macro features
  RandomForest  [B]      — housing history + FRED macro features
  HistGBT       [B]      — housing history + FRED macro features (gradient boosting)

Feature sets:
  A : zhvi_sfr_mid_dec, zhvi_sfr_mid_dec_prev, growth_prev
  B : A + mortgage_rate_avg, mortgage_rate_yoy, cpi_yoy_pct

  cpi_avg is intentionally excluded — it is a non-stationary time trend.

Leakage note:
  Macro features for YEAR=t describe conditions known by Dec 31 of year t.
  Target target_growth_1y covers Dec(t) → Dec(t+1). No future information used.

Best model selected by: highest directional_accuracy, tie-break lowest mae_pct.

Input  (read-only):
  Methoden Data/Modeling Data/Annual_Modeling_Table_v2_macro.csv

Outputs (new files only — app/data/ is not touched):
  app/data_v2_macro/metrics.csv
  app/data_v2_macro/predictions_test.csv
  app/data_v2_macro/investment_ranking.csv
  app/models/best_model_v2_macro.pkl
"""

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table_v2_macro.csv"
OUT_DIR = ROOT / "app" / "data_v2_macro"
MDL_DIR = ROOT / "app" / "models"

METRICS_OUT   = OUT_DIR / "metrics.csv"
PRED_TEST_OUT = OUT_DIR / "predictions_test.csv"
RANKING_OUT   = OUT_DIR / "investment_ranking.csv"
MODEL_OUT     = MDL_DIR / "best_model_v2_macro.pkl"

# v1 outputs for comparison (read-only)
V1_METRICS_PATH  = ROOT / "app" / "data" / "metrics.csv"
V1_PRED_PATH     = ROOT / "app" / "data" / "predictions_test.csv"

TARGET_COL = "target_growth_1y"

FEATURES_A = [
    "zhvi_sfr_mid_dec",
    "zhvi_sfr_mid_dec_prev",
    "growth_prev",
]

FEATURES_B = FEATURES_A + [
    "mortgage_rate_avg",
    "mortgage_rate_yoy",
    "cpi_yoy_pct",
]

# Leakage guard: cpi_avg must never appear as a feature (non-stationary index level)
assert "cpi_avg" not in FEATURES_B, "cpi_avg must not be used as a model feature"
assert all("next" not in f.lower() for f in FEATURES_B), "No 'next' columns allowed in features"

LABEL_THRESHOLDS = [
    ( 0.06,  float("inf"), "Attractive"),
    ( 0.02,  0.06,         "Moderate"),
    (-0.02,  0.02,         "Neutral"),
    (-float("inf"), -0.02, "Caution"),
]

SEP = "─" * 66


# ── helpers ────────────────────────────────────────────────────────────────────

def recommendation_label(growth: float) -> str:
    for lo, hi, label in LABEL_THRESHOLDS:
        if lo <= growth < hi:
            return label
    return "Caution"


def compute_metrics(y_true, y_pred, model_name: str, feature_set: str) -> dict:
    mae     = mean_absolute_error(y_true, y_pred) * 100
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2      = r2_score(y_true, y_pred)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        "model":                model_name,
        "feature_set":          feature_set,
        "mae_pct":              round(mae, 4),
        "rmse_pct":             round(rmse, 4),
        "r2":                   round(r2, 4),
        "directional_accuracy": round(dir_acc, 4),
    }


def build_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0)),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=300, max_features="sqrt",
            min_samples_leaf=5, random_state=42, n_jobs=-1,
        )),
    ])


def build_hgbt_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   HistGradientBoostingRegressor(
            max_iter=300, max_leaf_nodes=31,
            learning_rate=0.05, random_state=42,
        )),
    ])


def _fail(msg: str):
    print(f"\n[SANITY FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str):
    print(f"  [OK] {msg}")


# ── pre-flight ─────────────────────────────────────────────────────────────────

def pre_flight(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("PRE-FLIGHT SANITY CHECKS")
    print(SEP)

    # 1. shape
    if df.shape != (13211, 43):
        _fail(f"Unexpected shape {df.shape}; expected (13211, 43)")
    _ok(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. split counts
    sc = df["split"].value_counts().to_dict()
    expected = {"train": 9791, "test": 2565, "predict": 855}
    for sp, n in expected.items():
        if sc.get(sp, 0) != n:
            _fail(f"Split '{sp}': expected {n}, got {sc.get(sp, 0)}")
    _ok(f"Split counts: {sc}")

    # 3. required columns present
    required = (["CBSA_CODE", "CBSA_TITLE", "YEAR", "split", TARGET_COL]
                + FEATURES_B)
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(f"Missing columns: {missing}")
    _ok(f"All {len(required)} required columns present")

    # 4. target non-null in train/test
    for sp in ("train", "test"):
        n_null = df.loc[df["split"] == sp, TARGET_COL].isna().sum()
        if n_null > 0:
            _fail(f"{n_null} null targets in split='{sp}'")
    _ok("target_growth_1y non-null in train and test rows")

    # 5. predict target may be null
    pred_null  = df.loc[df["split"] == "predict", TARGET_COL].isna().sum()
    pred_total = (df["split"] == "predict").sum()
    _ok(f"predict rows: {pred_total} total, {pred_null} with null target (expected)")

    # 6. FEATURES_B non-null in train/test
    tt_df = df[df["split"].isin(["train", "test"])]
    mv = tt_df[FEATURES_B].isna().sum()
    total_mv = int(mv.sum())
    if total_mv > 0:
        _fail(f"FEATURES_B has {total_mv} nulls in train/test rows: {mv[mv>0].to_dict()}")
    _ok(f"FEATURES_B zero nulls in train+test rows")

    # 7. cpi_avg not a feature
    assert "cpi_avg" not in FEATURES_B
    _ok("cpi_avg correctly excluded from FEATURES_B (non-stationary — audit only)")

    # 8. no 'next' columns
    assert all("next" not in f.lower() for f in FEATURES_B)
    _ok("No 'next' columns in any feature set")

    # 9. year assignment
    if df.loc[df["split"] == "train", "YEAR"].max() > 2021:
        _fail("train set contains years after 2021")
    if not df.loc[df["split"] == "test", "YEAR"].between(2022, 2024).all():
        _fail("test set contains years outside 2022–2024")
    if not (df.loc[df["split"] == "predict", "YEAR"] == 2025).all():
        _fail("predict set contains years other than 2025")
    _ok("Year assignments correct: train ≤ 2021, test 2022–2024, predict = 2025")

    # 10. target summary stats
    print()
    print("  Target summary (target_growth_1y × 100):")
    for sp in ("train", "test"):
        t = df.loc[df["split"] == sp, TARGET_COL] * 100
        print(f"    {sp:6s}: n={len(t):,}  mean={t.mean():+.2f}%  "
              f"std={t.std():.2f}%  min={t.min():+.2f}%  max={t.max():+.2f}%  "
              f"pct_pos={(t>0).mean()*100:.1f}%")

    # 11. macro spike check: 2020–2025
    print()
    print("  Macro values 2020–2025 (confirm 2022 spike):")
    macro_check = (
        df[df["YEAR"].isin(range(2020, 2026))]
        [["YEAR", "mortgage_rate_avg", "mortgage_rate_yoy", "cpi_yoy_pct"]]
        .drop_duplicates("YEAR")
        .sort_values("YEAR")
    )
    print(macro_check.to_string(index=False))
    yoy_2022 = float(macro_check.loc[macro_check["YEAR"] == 2022, "mortgage_rate_yoy"].iloc[0])
    if yoy_2022 > 1.0:
        print(f"  ✓ mortgage_rate_yoy 2022 = {yoy_2022:+.4f} pp — rate shock visible in features")
    else:
        _fail(f"mortgage_rate_yoy 2022 = {yoy_2022:.4f} — expected > +1.0 pp")

    print()


# ── ranking backtest ───────────────────────────────────────────────────────────

def ranking_backtest(pred_df: pd.DataFrame, label: str, top_pct: float = 0.10):
    """Print top/all/bottom spread, hit rate, and Spearman ρ."""
    print(f"\n  Ranking backtest — {label}:")
    rows = []
    for yr, g in pred_df.groupby("YEAR"):
        k   = max(1, math.ceil(len(g) * top_pct))
        gs  = g.sort_values("predicted_growth", ascending=False)
        top = gs.head(k)
        bot = gs.tail(k)
        rows.append({
            "Year":             int(yr),
            "Top 10% Mean (%)": round(top["actual_growth"].mean(), 2),
            "All Mean (%)":     round(g["actual_growth"].mean(), 2),
            "Bot 10% Mean (%)": round(bot["actual_growth"].mean(), 2),
            "Spread (pp)":      round(top["actual_growth"].mean()
                                      - bot["actual_growth"].mean(), 2),
            "Hit Rate (%)":     round((top["actual_growth"] > 0).mean() * 100, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # pooled
    k_all   = max(1, math.ceil(len(pred_df) * top_pct))
    ranked  = pred_df.sort_values("predicted_growth", ascending=False)
    top_all = ranked.head(k_all)
    bot_all = ranked.tail(k_all)
    ov_spr  = round(top_all["actual_growth"].mean() - bot_all["actual_growth"].mean(), 2)
    ov_hit  = round((top_all["actual_growth"] > 0).mean() * 100, 1)
    rho     = pred_df["predicted_growth"].rank().corr(pred_df["actual_growth"].rank())
    print(f"\n  Pooled (n={len(pred_df):,}): "
          f"top10% mean={top_all['actual_growth'].mean():+.2f}%  "
          f"all mean={pred_df['actual_growth'].mean():+.2f}%  "
          f"bot10% mean={bot_all['actual_growth'].mean():+.2f}%  "
          f"spread={ov_spr:+.2f} pp  hit={ov_hit:.1f}%  "
          f"Spearman ρ={rho:.4f}")
    return rho


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    if not IN_FILE.exists():
        print(f"[ERROR] Input not found: {IN_FILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(IN_FILE, dtype={"CBSA_CODE": str})
    pre_flight(df)

    # ── split ─────────────────────────────────────────────────────────────────
    train_df   = df[df["split"] == "train"].copy()
    test_df    = df[df["split"] == "test"].copy()
    predict_df = df[df["split"] == "predict"].copy()

    y_train = train_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # ── train ─────────────────────────────────────────────────────────────────
    print(SEP)
    print("TRAINING")
    print(SEP)

    models = [
        ("Ridge",        "A", build_ridge_pipeline(),  FEATURES_A),
        ("Ridge",        "B", build_ridge_pipeline(),  FEATURES_B),
        ("RandomForest", "B", build_rf_pipeline(),     FEATURES_B),
        ("HistGBT",      "B", build_hgbt_pipeline(),   FEATURES_B),
    ]

    all_metrics  = []
    trained_models = {}

    naive_val   = float(np.median(y_train))
    naive_preds = np.full_like(y_test, fill_value=naive_val, dtype=float)
    naive_m     = compute_metrics(y_test, naive_preds, "Naive baseline", "—")
    all_metrics.append(naive_m)
    print(f"  Naive baseline  (median={naive_val*100:+.2f}%)  "
          f"dir_acc={naive_m['directional_accuracy']:.4f}  mae={naive_m['mae_pct']:.4f}")

    for model_name, fs_name, pipe, features in models:
        X_train = train_df[features].values
        X_test  = test_df[features].values
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = compute_metrics(y_test, preds, model_name, fs_name)
        all_metrics.append(m)
        trained_models[(model_name, fs_name)] = (pipe, features)
        print(f"  {model_name:14s} [{fs_name}]  "
              f"dir_acc={m['directional_accuracy']:.4f}  "
              f"mae={m['mae_pct']:.4f}  rmse={m['rmse_pct']:.4f}  r2={m['r2']:.4f}")

    # ── select best ───────────────────────────────────────────────────────────
    metrics_df  = pd.DataFrame(all_metrics)
    candidates  = (metrics_df[metrics_df["model"] != "Naive baseline"]
                   .sort_values(["directional_accuracy", "mae_pct"],
                                ascending=[False, True])
                   .reset_index(drop=True))
    best_row            = candidates.iloc[0]
    best_key            = (best_row["model"], best_row["feature_set"])
    best_pipe, best_feats = trained_models[best_key]

    print(f"\n  → Best v2 model: {best_row['model']} [{best_row['feature_set']}]  "
          f"dir_acc={best_row['directional_accuracy']:.4f}  "
          f"mae={best_row['mae_pct']:.4f} pp")

    # ── save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MDL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_OUT, index=False)

    # predictions_test
    preds_test = best_pipe.predict(test_df[best_feats].values)
    pred_test_df = pd.DataFrame({
        "CBSA_CODE":         test_df["CBSA_CODE"].values,
        "CBSA_TITLE":        test_df["CBSA_TITLE"].values,
        "YEAR":              test_df["YEAR"].values,
        "actual_growth":     np.round(y_test * 100, 4),
        "predicted_growth":  np.round(preds_test * 100, 4),
        "abs_error":         np.round(np.abs(y_test - preds_test) * 100, 4),
        "direction_correct": (np.sign(preds_test) == np.sign(y_test)).astype(int),
    })
    pred_test_df.to_csv(PRED_TEST_OUT, index=False)

    # investment_ranking
    preds_2026 = best_pipe.predict(predict_df[best_feats].values)
    ranking_df = pd.DataFrame({
        "CBSA_CODE":             predict_df["CBSA_CODE"].values,
        "CBSA_TITLE":            predict_df["CBSA_TITLE"].values,
        "YEAR":                  predict_df["YEAR"].values,
        "predicted_growth_2026": np.round(preds_2026 * 100, 4),
        "recommendation_label":  [recommendation_label(g) for g in preds_2026],
        "zhvi_sfr_mid_dec":      predict_df["zhvi_sfr_mid_dec"].values,
        "mortgage_rate_avg":     predict_df["mortgage_rate_avg"].values,
        "mortgage_rate_yoy":     predict_df["mortgage_rate_yoy"].values,
        "cpi_yoy_pct":           predict_df["cpi_yoy_pct"].values,
    })
    for opt in ("had_major_disaster_ann", "damage_property_sum_ann"):
        if opt in predict_df.columns:
            ranking_df[opt] = predict_df[opt].values
    ranking_df = (ranking_df.sort_values("predicted_growth_2026", ascending=False)
                             .reset_index(drop=True))
    ranking_df.to_csv(RANKING_OUT, index=False)

    joblib.dump({"pipeline": best_pipe, "features": best_feats, "name": best_key}, MODEL_OUT)

    # ── post-training comparison ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("POST-TRAINING COMPARISON")
    print(SEP)

    # v2 metrics
    print("\n  v2 metrics table:")
    print(metrics_df[["model", "feature_set", "mae_pct", "rmse_pct", "r2",
                       "directional_accuracy"]].to_string(index=False))

    print(f"\n  Best v2 model  : {best_row['model']} [{best_row['feature_set']}]")
    print(f"  Selection reason: highest directional_accuracy={best_row['directional_accuracy']:.4f}, "
          f"mae={best_row['mae_pct']:.4f} pp")

    # side-by-side v1 vs v2 (for all models appearing in v1)
    if V1_METRICS_PATH.exists():
        v1_m = pd.read_csv(V1_METRICS_PATH)
        print(f"\n  Side-by-side v1 vs v2 — directional accuracy and MAE:")
        header = f"  {'Model':<18} {'FS':<7} {'dir_acc_v1':>10} {'dir_acc_v2':>10}  {'Δ dir_acc':>9}  {'mae_v1':>8} {'mae_v2':>8}  {'Δ mae':>8}"
        print(header)
        print(f"  {'-'*90}")
        # show rows present in v1
        for _, v1r in v1_m.iterrows():
            v2r = metrics_df[
                (metrics_df["model"] == v1r["model"]) &
                (metrics_df["feature_set"] == v1r["feature_set"])
            ]
            da_v1  = v1r["directional_accuracy"]
            mae_v1 = v1r["mae_pct"]
            if len(v2r):
                da_v2  = v2r.iloc[0]["directional_accuracy"]
                mae_v2 = v2r.iloc[0]["mae_pct"]
                print(f"  {v1r['model']:<18} {v1r['feature_set']:<7} "
                      f"{da_v1:>10.4f} {da_v2:>10.4f}  {da_v2-da_v1:>+9.4f}  "
                      f"{mae_v1:>8.4f} {mae_v2:>8.4f}  {mae_v2-mae_v1:>+8.4f}")
            else:
                print(f"  {v1r['model']:<18} {v1r['feature_set']:<7} "
                      f"{da_v1:>10.4f} {'—':>10}  {'—':>9}  {mae_v1:>8.4f} {'—':>8}  {'—':>8}")
        # show v2-only rows
        for _, v2r in metrics_df.iterrows():
            in_v1 = ((v1_m["model"] == v2r["model"]) &
                     (v1_m["feature_set"] == v2r["feature_set"])).any()
            if not in_v1:
                print(f"  {v2r['model']:<18} {v2r['feature_set']:<7} "
                      f"{'—':>10} {v2r['directional_accuracy']:>10.4f}  {'NEW':>9}  "
                      f"{'—':>8} {v2r['mae_pct']:>8.4f}  {'NEW':>8}")
    else:
        print("  [WARN] v1 metrics.csv not found — skipping side-by-side")

    # systematic bias check
    actual_mean = float(np.mean(y_test) * 100)
    v2_pred_mean = float(np.mean(preds_test) * 100)
    v2_bias = v2_pred_mean - actual_mean

    print(f"\n  Systematic bias check:")
    print(f"    actual test mean growth    : {actual_mean:+.4f}%")

    if V1_PRED_PATH.exists():
        v1_pred_df  = pd.read_csv(V1_PRED_PATH)
        v1_pred_mean = float(v1_pred_df["predicted_growth"].mean())
        v1_bias      = v1_pred_mean - actual_mean
        print(f"    v1 predicted mean (Ridge A): {v1_pred_mean:+.4f}%")
        print(f"    v2 predicted mean ({best_row['model']} {best_row['feature_set']}): {v2_pred_mean:+.4f}%")
        print(f"    v1 bias            : {v1_bias:+.4f} pp")
        print(f"    v2 bias            : {v2_bias:+.4f} pp")
        bias_delta = abs(v2_bias) - abs(v1_bias)
        if bias_delta < 0:
            print(f"    ✓ Bias shrank by {abs(bias_delta):.4f} pp  (|v2| < |v1|)")
        elif bias_delta == 0:
            print(f"    = Bias unchanged")
        else:
            print(f"    ✗ Bias grew by {bias_delta:.4f} pp  (|v2| > |v1|)")
    else:
        print(f"    v2 predicted mean          : {v2_pred_mean:+.4f}%")
        print(f"    v2 bias                    : {v2_bias:+.4f} pp")
        print("    [WARN] v1 predictions.csv not found — v1 bias not computed")

    # ranking backtest — v2
    rho_v2 = ranking_backtest(pred_test_df, label=f"v2 {best_row['model']} [{best_row['feature_set']}]")

    # ranking backtest — v1 for comparison
    if V1_PRED_PATH.exists():
        rho_v1 = ranking_backtest(v1_pred_df, label="v1 Ridge [A]")
        print(f"\n  Spearman ρ comparison: v1={rho_v1:.4f}  v2={rho_v2:.4f}  "
              f"Δ={rho_v2-rho_v1:+.4f}")

    # ── output existence check ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("OUTPUT FILE CHECK")
    print(SEP)
    for path in (METRICS_OUT, PRED_TEST_OUT, RANKING_OUT, MODEL_OUT):
        exists = path.exists()
        print(f"  {'✓' if exists else '✗'} {path.relative_to(ROOT)}")
        if not exists:
            _fail(f"Output missing: {path}")

    print()
    print(f"  v2 metrics shape       : {metrics_df.shape}")
    print(f"  predictions_test shape : {pred_test_df.shape}")
    print(f"  investment_ranking rows: {len(ranking_df)}")

    # label distribution
    lbl_dist = ranking_df["recommendation_label"].value_counts().to_dict()
    print(f"  Label distribution     : {lbl_dist}")

    # NaN checks
    if ranking_df["predicted_growth_2026"].isna().any():
        _fail("NaN predictions in investment_ranking")
    if pred_test_df["predicted_growth"].isna().any():
        _fail("NaN predictions in predictions_test")
    _ok("No NaN predictions in either output file")

    print(f"\n{SEP}")
    print("DONE — Phase V2.2 complete")
    print(SEP)


if __name__ == "__main__":
    main()
