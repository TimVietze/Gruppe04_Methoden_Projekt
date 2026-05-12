"""
03_ml_v2_macro_laus.py  —  Phase V2.4

scikit-learn ML prototype adding CBSA-level BLS LAUS unemployment features
to the FRED national macro feature set.

Research question:
  Does CBSA-level unemployment (BLS LAUS) improve directional accuracy and
  ranking value beyond housing history alone (v1) and beyond national macro
  features (v2_macro)?

Models trained:
  Naive baseline                — predict median training target for every row
  Ridge         [A]             — housing history only (v1 baseline reproduced)
  Ridge         [B]             — housing history + FRED macro
  Ridge         [C]             — housing + macro + LAUS unemployment
  Lasso         [C]             — housing + macro + LAUS unemployment
  RandomForest  [C]             — housing + macro + LAUS unemployment
  HistGBT       [C]             — housing + macro + LAUS unemployment

Feature sets:
  A : zhvi_sfr_mid_dec, zhvi_sfr_mid_dec_prev, growth_prev
  B : A + mortgage_rate_avg, mortgage_rate_yoy, cpi_yoy_pct
  C : B + unemployment_rate, unemployment_yoy, unemployment_vs_national

  cpi_avg intentionally excluded — non-stationary index level, audit only.

Leakage note:
  All features for year t are fully known by Dec 31 of year t.
  Target target_growth_1y covers Dec(t) → Dec(t+1). No future information used.

Best model selected by: highest directional_accuracy, tie-break lowest mae_pct.

Input  (read-only):
  Methoden Data/Modeling Data/Annual_Modeling_Table_v2_macro_laus.csv

Comparison inputs (read-only):
  app/data/metrics.csv               (v1 — housing only)
  app/data/predictions_test.csv
  app/data_v2_macro/metrics.csv      (v2 — national macro)
  app/data_v2_macro/predictions_test.csv

Outputs (new files only — existing outputs are not touched):
  app/data_v2_macro_laus/metrics.csv
  app/data_v2_macro_laus/predictions_test.csv
  app/data_v2_macro_laus/investment_ranking.csv
  app/models/best_model_v2_macro_laus.pkl
"""

import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table_v2_macro_laus.csv"
OUT_DIR = ROOT / "app" / "data_v2_macro_laus"
MDL_DIR = ROOT / "app" / "models"

METRICS_OUT   = OUT_DIR / "metrics.csv"
PRED_TEST_OUT = OUT_DIR / "predictions_test.csv"
RANKING_OUT   = OUT_DIR / "investment_ranking.csv"
MODEL_OUT     = MDL_DIR / "best_model_v2_macro_laus.pkl"

# comparison inputs (read-only)
V1_METRICS_PATH   = ROOT / "app" / "data" / "metrics.csv"
V1_PRED_PATH      = ROOT / "app" / "data" / "predictions_test.csv"
V2M_METRICS_PATH  = ROOT / "app" / "data_v2_macro" / "metrics.csv"
V2M_PRED_PATH     = ROOT / "app" / "data_v2_macro" / "predictions_test.csv"

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

FEATURES_C = FEATURES_B + [
    "unemployment_rate",
    "unemployment_yoy",
    "unemployment_vs_national",
]

# Leakage guards — fail at import time if violated
assert "cpi_avg" not in FEATURES_C, "cpi_avg must not be a model feature"
assert all("next" not in f.lower() for f in FEATURES_C), "No 'next' columns allowed"
assert TARGET_COL not in FEATURES_C, "Target must not appear in feature set"

LABEL_THRESHOLDS = [
    ( 0.06,  float("inf"), "Attractive"),
    ( 0.02,  0.06,         "Moderate"),
    (-0.02,  0.02,         "Neutral"),
    (-float("inf"), -0.02, "Caution"),
]

SEP = "─" * 68


# ── helpers ────────────────────────────────────────────────────────────────────

def recommendation_label(growth: float) -> str:
    for lo, hi, label in LABEL_THRESHOLDS:
        if lo <= growth < hi:
            return label
    return "Caution"


def compute_metrics(y_true, y_pred, model_name: str, feature_set: str) -> dict:
    mae    = mean_absolute_error(y_true, y_pred) * 100
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2     = r2_score(y_true, y_pred)
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


def build_lasso_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Lasso(alpha=0.01, max_iter=5000)),
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


def ranking_backtest(pred_df: pd.DataFrame, label: str, top_pct: float = 0.10):
    """Print per-year and pooled top/bottom spread, hit rate, and Spearman ρ."""
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

    k_all   = max(1, math.ceil(len(pred_df) * top_pct))
    ranked  = pred_df.sort_values("predicted_growth", ascending=False)
    top_all = ranked.head(k_all)
    bot_all = ranked.tail(k_all)
    spread  = round(top_all["actual_growth"].mean() - bot_all["actual_growth"].mean(), 2)
    hit     = round((top_all["actual_growth"] > 0).mean() * 100, 1)
    rho     = pred_df["predicted_growth"].rank().corr(pred_df["actual_growth"].rank())
    print(f"\n  Pooled (n={len(pred_df):,}): "
          f"top10% mean={top_all['actual_growth'].mean():+.2f}%  "
          f"all mean={pred_df['actual_growth'].mean():+.2f}%  "
          f"bot10% mean={bot_all['actual_growth'].mean():+.2f}%  "
          f"spread={spread:+.2f} pp  hit={hit:.1f}%  "
          f"Spearman ρ={rho:.4f}")
    return rho, spread, hit


# ── pre-flight ─────────────────────────────────────────────────────────────────

def pre_flight(df: pd.DataFrame):
    print(f"\n{SEP}")
    print("PRE-FLIGHT SANITY CHECKS")
    print(SEP)

    # 1. shape
    if df.shape != (13211, 46):
        _fail(f"Unexpected shape {df.shape}; expected (13211, 46)")
    _ok(f"Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. split counts
    sc = df["split"].value_counts().to_dict()
    expected = {"train": 9791, "test": 2565, "predict": 855}
    for sp, n in expected.items():
        if sc.get(sp, 0) != n:
            _fail(f"Split '{sp}': expected {n}, got {sc.get(sp, 0)}")
    _ok(f"Split counts: train={sc['train']:,}  test={sc['test']:,}  predict={sc['predict']:,}")

    # 3. all feature columns present
    all_features = sorted(set(FEATURES_A + FEATURES_B + FEATURES_C))
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        _fail(f"Missing feature columns: {missing}")
    _ok(f"All {len(all_features)} feature columns present "
        f"(A={len(FEATURES_A)}, B={len(FEATURES_B)}, C={len(FEATURES_C)})")

    # 4. target non-null in train/test
    for sp in ("train", "test"):
        n_null = df.loc[df["split"] == sp, TARGET_COL].isna().sum()
        if n_null > 0:
            _fail(f"{n_null} null targets in split='{sp}'")
    _ok("target_growth_1y non-null in all train and test rows")

    # 5. predict target may be null
    pred_null  = df.loc[df["split"] == "predict", TARGET_COL].isna().sum()
    pred_total = (df["split"] == "predict").sum()
    _ok(f"predict rows: {pred_total} total, {pred_null} with null target (expected)")

    # 6. null counts per feature set in train+test
    tt_df = df[df["split"].isin(["train", "test"])]
    print()
    print("  Null counts in train+test rows by feature set:")
    for fs_name, fs in [("A", FEATURES_A), ("B", FEATURES_B), ("C", FEATURES_C)]:
        mv = tt_df[fs].isna().sum()
        total = int(mv.sum())
        if total > 0:
            _fail(f"Feature set {fs_name} has {total} nulls in train+test: {mv[mv>0].to_dict()}")
        print(f"    Feature set {fs_name} ({len(fs)} features): 0 nulls  [OK]")

    # 7. cpi_avg not a feature
    assert "cpi_avg" not in FEATURES_C
    _ok("cpi_avg excluded from all feature sets (non-stationary — audit column only)")

    # 8. no 'next' columns
    assert all("next" not in f.lower() for f in FEATURES_C)
    _ok("No 'next' columns in any feature set")

    # 9. year assignment
    if df.loc[df["split"] == "train", "YEAR"].max() > 2021:
        _fail("train set contains years after 2021")
    if not df.loc[df["split"] == "test", "YEAR"].between(2022, 2024).all():
        _fail("test set contains years outside 2022–2024")
    if not (df.loc[df["split"] == "predict", "YEAR"] == 2025).all():
        _fail("predict set contains years other than 2025")
    _ok("Year assignments: train ≤ 2021  |  test 2022–2024  |  predict = 2025")

    # 10. target summary
    print()
    print("  Target summary (target_growth_1y × 100):")
    for sp in ("train", "test"):
        t = df.loc[df["split"] == sp, TARGET_COL] * 100
        print(f"    {sp:7s}: n={len(t):,}  mean={t.mean():+.2f}%  "
              f"std={t.std():.2f}%  min={t.min():+.2f}%  max={t.max():+.2f}%  "
              f"pct_pos={(t>0).mean()*100:.1f}%")

    # 11. LAUS unemployment spike check (2019–2024)
    print()
    print("  LAUS unemployment check 2019–2024:")
    unemp = (
        df[df["YEAR"].isin(range(2019, 2025))]
        .groupby("YEAR")["unemployment_rate"]
        .agg(cbsa_mean="mean", cbsa_min="min", cbsa_max="max")
        .round(3)
        .reset_index()
    )
    print(unemp.to_string(index=False))
    mean_2020 = float(unemp.loc[unemp["YEAR"] == 2020, "cbsa_mean"].iloc[0])
    mean_2019 = float(unemp.loc[unemp["YEAR"] == 2019, "cbsa_mean"].iloc[0])
    if mean_2020 > mean_2019 + 1.0:
        print(f"  ✓ COVID spike: 2020 cbsa_mean={mean_2020:.3f} vs 2019={mean_2019:.3f} — confirmed")
    else:
        _fail(f"COVID spike not detected in unemployment (2020={mean_2020:.3f}, 2019={mean_2019:.3f})")

    print()


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

    model_specs = [
        ("Ridge",        "A", build_ridge_pipeline(),  FEATURES_A),
        ("Ridge",        "B", build_ridge_pipeline(),  FEATURES_B),
        ("Ridge",        "C", build_ridge_pipeline(),  FEATURES_C),
        ("Lasso",        "C", build_lasso_pipeline(),  FEATURES_C),
        ("RandomForest", "C", build_rf_pipeline(),     FEATURES_C),
        ("HistGBT",      "C", build_hgbt_pipeline(),   FEATURES_C),
    ]

    all_metrics: list[dict] = []
    trained_models: dict = {}

    naive_val   = float(np.median(y_train))
    naive_preds = np.full_like(y_test, fill_value=naive_val, dtype=float)
    naive_m     = compute_metrics(y_test, naive_preds, "Naive baseline", "—")
    all_metrics.append(naive_m)
    print(f"  {'Naive baseline':<18} [{'—':2}]  "
          f"dir_acc={naive_m['directional_accuracy']:.4f}  "
          f"mae={naive_m['mae_pct']:.4f}  "
          f"(median={naive_val*100:+.2f}%)")

    for model_name, fs_name, pipe, features in model_specs:
        X_train = train_df[features].values
        X_test  = test_df[features].values
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = compute_metrics(y_test, preds, model_name, fs_name)
        all_metrics.append(m)
        trained_models[(model_name, fs_name)] = (pipe, features)
        print(f"  {model_name:<18} [{fs_name:2}]  "
              f"dir_acc={m['directional_accuracy']:.4f}  "
              f"mae={m['mae_pct']:.4f}  "
              f"rmse={m['rmse_pct']:.4f}  "
              f"r2={m['r2']:.4f}")

    # ── select best ───────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    candidates = (
        metrics_df[metrics_df["model"] != "Naive baseline"]
        .sort_values(["directional_accuracy", "mae_pct"], ascending=[False, True])
        .reset_index(drop=True)
    )
    best_row              = candidates.iloc[0]
    best_key              = (best_row["model"], best_row["feature_set"])
    best_pipe, best_feats = trained_models[best_key]

    print(f"\n  → Best v3 model: {best_row['model']} [{best_row['feature_set']}]  "
          f"dir_acc={best_row['directional_accuracy']:.4f}  "
          f"mae={best_row['mae_pct']:.4f} pp")

    # ── save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MDL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(METRICS_OUT, index=False)

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
        "unemployment_rate":     predict_df["unemployment_rate"].values,
        "unemployment_yoy":      predict_df["unemployment_yoy"].values,
        "unemployment_vs_national": predict_df["unemployment_vs_national"].values,
    })
    for opt in ("had_major_disaster_ann", "damage_property_sum_ann"):
        if opt in predict_df.columns:
            ranking_df[opt] = predict_df[opt].values
    ranking_df = (ranking_df.sort_values("predicted_growth_2026", ascending=False)
                             .reset_index(drop=True))
    ranking_df.to_csv(RANKING_OUT, index=False)

    joblib.dump({"pipeline": best_pipe, "features": best_feats, "name": best_key}, MODEL_OUT)

    # ── metrics table ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("TRAINING METRICS — ALL MODELS")
    print(SEP)
    print(metrics_df[["model", "feature_set", "mae_pct", "rmse_pct", "r2",
                       "directional_accuracy"]].to_string(index=False))

    # ── comparison vs v1 and v2_macro ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("COMPARISON VS V1 AND V2_MACRO")
    print(SEP)

    def load_safe(path: Path) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_csv(path)
        print(f"  [WARN] Not found: {path.relative_to(ROOT)}")
        return None

    v1_m   = load_safe(V1_METRICS_PATH)
    v1_p   = load_safe(V1_PRED_PATH)
    v2m_m  = load_safe(V2M_METRICS_PATH)
    v2m_p  = load_safe(V2M_PRED_PATH)

    # Full directional accuracy comparison table
    print("\n  Directional accuracy by model:")
    header = (f"  {'Model':<18} {'FS':<4}  {'v1':>8}  {'v2_macro':>8}  {'v3_laus':>8}  "
              f"{'Δ v3-v1':>9}  {'Δ v3-v2m':>9}")
    print(header)
    print(f"  {'─'*78}")

    all_v3 = {(r["model"], r["feature_set"]): r for _, r in metrics_df.iterrows()}
    all_v1  = ({(r["model"], r["feature_set"]): r for _, r in v1_m.iterrows()}
               if v1_m is not None else {})
    all_v2m = ({(r["model"], r["feature_set"]): r for _, r in v2m_m.iterrows()}
               if v2m_m is not None else {})

    for _, v3r in metrics_df.iterrows():
        key = (v3r["model"], v3r["feature_set"])
        da_v3  = v3r["directional_accuracy"]
        da_v1  = all_v1[key]["directional_accuracy"]  if key in all_v1  else None
        da_v2m = all_v2m[key]["directional_accuracy"] if key in all_v2m else None
        d_v1   = f"{da_v3 - da_v1:+.4f}"  if da_v1  is not None else "   NEW"
        d_v2m  = f"{da_v3 - da_v2m:+.4f}" if da_v2m is not None else "   NEW"
        sv1    = f"{da_v1:.4f}"  if da_v1  is not None else "       —"
        sv2m   = f"{da_v2m:.4f}" if da_v2m is not None else "       —"
        print(f"  {v3r['model']:<18} {v3r['feature_set']:<4}  "
              f"{sv1:>8}  {sv2m:>8}  {da_v3:>8.4f}  {d_v1:>9}  {d_v2m:>9}")

    # MAE comparison table
    print("\n  MAE (pp) by model:")
    header2 = (f"  {'Model':<18} {'FS':<4}  {'v1':>7}  {'v2_macro':>8}  {'v3_laus':>8}  "
               f"{'Δ v3-v1':>9}  {'Δ v3-v2m':>9}")
    print(header2)
    print(f"  {'─'*78}")
    for _, v3r in metrics_df.iterrows():
        key = (v3r["model"], v3r["feature_set"])
        mae_v3  = v3r["mae_pct"]
        mae_v1  = all_v1[key]["mae_pct"]  if key in all_v1  else None
        mae_v2m = all_v2m[key]["mae_pct"] if key in all_v2m else None
        d_v1    = f"{mae_v3 - mae_v1:+.4f}"  if mae_v1  is not None else "   NEW"
        d_v2m   = f"{mae_v3 - mae_v2m:+.4f}" if mae_v2m is not None else "   NEW"
        sv1     = f"{mae_v1:.4f}"  if mae_v1  is not None else "      —"
        sv2m    = f"{mae_v2m:.4f}" if mae_v2m is not None else "       —"
        print(f"  {v3r['model']:<18} {v3r['feature_set']:<4}  "
              f"{sv1:>7}  {sv2m:>8}  {mae_v3:>8.4f}  {d_v1:>9}  {d_v2m:>9}")

    # Bias check
    actual_mean  = float(np.mean(y_test) * 100)
    v3_pred_mean = float(np.mean(preds_test) * 100)
    v3_bias      = v3_pred_mean - actual_mean

    print(f"\n  Predicted mean bias (predicted_mean − actual_mean):")
    print(f"    actual test mean growth        : {actual_mean:+.4f}%")
    if v1_p is not None:
        v1_pred_mean = float(v1_p["predicted_growth"].mean())
        v1_bias      = v1_pred_mean - actual_mean
        print(f"    v1  predicted mean (best model): {v1_pred_mean:+.4f}%  bias={v1_bias:+.4f} pp")
    else:
        v1_bias = None
    if v2m_p is not None:
        v2m_pred_mean = float(v2m_p["predicted_growth"].mean())
        v2m_bias      = v2m_pred_mean - actual_mean
        print(f"    v2m predicted mean (best model): {v2m_pred_mean:+.4f}%  bias={v2m_bias:+.4f} pp")
    else:
        v2m_bias = None
    print(f"    v3  predicted mean (best model): {v3_pred_mean:+.4f}%  bias={v3_bias:+.4f} pp")

    if v1_bias is not None:
        delta = abs(v3_bias) - abs(v1_bias)
        mark  = "✓" if delta < -0.01 else ("=" if abs(delta) <= 0.01 else "✗")
        print(f"    |v3 bias| vs |v1 bias|: {mark}  Δ={delta:+.4f} pp")

    # ── ranking backtests ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("RANKING BACKTEST — TEST YEARS 2022–2024")
    print(SEP)

    rho_v3, spread_v3, hit_v3 = ranking_backtest(
        pred_test_df,
        label=f"v3 {best_row['model']} [{best_row['feature_set']}]",
    )

    rho_v1 = rho_v1_spread = rho_v1_hit = None
    if v1_p is not None:
        rho_v1, spread_v1, hit_v1 = ranking_backtest(v1_p, label="v1 Ridge [A]")
        rho_v1_spread = spread_v1
        rho_v1_hit    = hit_v1

    rho_v2m = rho_v2m_spread = rho_v2m_hit = None
    if v2m_p is not None:
        rho_v2m, spread_v2m, hit_v2m = ranking_backtest(v2m_p, label="v2_macro best")
        rho_v2m_spread = spread_v2m
        rho_v2m_hit    = hit_v2m

    print(f"\n  Spearman ρ summary:")
    if rho_v1 is not None:
        print(f"    v1            : ρ={rho_v1:.4f}  spread={rho_v1_spread:+.2f} pp  hit={rho_v1_hit:.1f}%")
    if rho_v2m is not None:
        print(f"    v2_macro      : ρ={rho_v2m:.4f}  spread={rho_v2m_spread:+.2f} pp  hit={rho_v2m_hit:.1f}%")
    print(f"    v3 (LAUS)     : ρ={rho_v3:.4f}  spread={spread_v3:+.2f} pp  hit={hit_v3:.1f}%")
    if rho_v1 is not None:
        print(f"    Δ v3 − v1     : ρ Δ={rho_v3 - rho_v1:+.4f}  "
              f"spread Δ={spread_v3 - rho_v1_spread:+.2f} pp  "
              f"hit Δ={hit_v3 - rho_v1_hit:+.1f}%")

    # ── verdict ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("VERDICT — DOES BLS LAUS UNEMPLOYMENT IMPROVE THE MODEL?")
    print(SEP)

    best_v3_da   = float(best_row["directional_accuracy"])
    best_v3_mae  = float(best_row["mae_pct"])
    best_v3_rmse = float(best_row["rmse_pct"])
    best_v3_r2   = float(best_row["r2"])

    # Identify v1 Ridge[A] row for exact comparison
    v1_ridge_a_da = v1_ridge_a_mae = v1_ridge_a_rmse = v1_ridge_a_r2 = None
    if v1_m is not None:
        rA = v1_m[(v1_m["model"] == "Ridge") & (v1_m["feature_set"] == "A")]
        if len(rA):
            v1_ridge_a_da   = float(rA.iloc[0]["directional_accuracy"])
            v1_ridge_a_mae  = float(rA.iloc[0]["mae_pct"])
            v1_ridge_a_rmse = float(rA.iloc[0]["rmse_pct"])
            v1_ridge_a_r2   = float(rA.iloc[0]["r2"])

    def verdict_line(question: str, result_str: str, positive: bool):
        mark = "✓" if positive else "✗"
        print(f"\n  {mark} {question}")
        print(f"      {result_str}")

    # Q1: exact forecast accuracy
    if v1_ridge_a_mae is not None:
        mae_delta = best_v3_mae - v1_ridge_a_mae
        r2_delta  = best_v3_r2  - v1_ridge_a_r2
        improved_acc = mae_delta < -0.05
        verdict_line(
            "Did LAUS improve exact forecast accuracy? (MAE, RMSE, R²)",
            (f"v3 best MAE={best_v3_mae:.4f} pp vs v1 Ridge[A] MAE={v1_ridge_a_mae:.4f} pp  "
             f"Δ={mae_delta:+.4f} pp  |  "
             f"R² v3={best_v3_r2:.4f} vs v1={v1_ridge_a_r2:.4f}  Δ={r2_delta:+.4f}"),
            improved_acc,
        )
    else:
        print("\n  ? Did LAUS improve exact forecast accuracy?  (v1 metrics unavailable)")

    # Q2: directional / ranking value
    if v1_ridge_a_da is not None:
        da_delta      = best_v3_da - v1_ridge_a_da
        rho_delta     = (rho_v3 - rho_v1) if rho_v1 is not None else None
        improved_dir  = da_delta > 0.001
        rho_str       = f"  Spearman ρ Δ={rho_delta:+.4f}" if rho_delta is not None else ""
        verdict_line(
            "Did LAUS improve directional / ranking value?",
            (f"dir_acc v3={best_v3_da:.4f} vs v1={v1_ridge_a_da:.4f}  "
             f"Δ={da_delta:+.4f}{rho_str}"),
            improved_dir,
        )
    else:
        print("\n  ? Did LAUS improve directional / ranking value?  (v1 metrics unavailable)")

    # Q3: systematic bias
    if v1_bias is not None:
        bias_delta   = abs(v3_bias) - abs(v1_bias)
        bias_reduced = bias_delta < -0.1
        verdict_line(
            "Did LAUS reduce systematic over-prediction bias?",
            (f"|v3 bias|={abs(v3_bias):.4f} pp vs |v1 bias|={abs(v1_bias):.4f} pp  "
             f"Δ={bias_delta:+.4f} pp"),
            bias_reduced,
        )
    else:
        print(f"\n  ? Systematic bias: v3 bias={v3_bias:+.4f} pp  (v1 bias unavailable)")

    # Q4: strong enough to replace v1 app outputs?
    if v1_ridge_a_da is not None and rho_v1 is not None:
        meaningful_da  = da_delta >= 0.005
        meaningful_rho = (rho_v3 - rho_v1) >= 0.005
        replace_worthy = meaningful_da or meaningful_rho
        verdict_line(
            "Is the improvement strong enough to replace v1 app outputs?",
            (f"dir_acc gain={da_delta:+.4f}  ρ gain={rho_v3-rho_v1:+.4f}  "
             f"threshold ≥ 0.005 on either metric"),
            replace_worthy,
        )
    else:
        print("\n  ? Replace v1? — comparison data unavailable")

    # ── output file check ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("OUTPUT FILE CHECK")
    print(SEP)
    all_ok = True
    for path in (METRICS_OUT, PRED_TEST_OUT, RANKING_OUT, MODEL_OUT):
        exists = path.exists()
        mark   = "✓" if exists else "✗"
        print(f"  {mark} {path.relative_to(ROOT)}")
        if not exists:
            all_ok = False
    if not all_ok:
        _fail("One or more output files missing — check errors above")

    print()
    print(f"  metrics.csv shape          : {metrics_df.shape}")
    print(f"  predictions_test shape     : {pred_test_df.shape}")
    print(f"  investment_ranking rows    : {len(ranking_df)}")

    lbl_dist = ranking_df["recommendation_label"].value_counts().to_dict()
    print(f"  Label distribution         : {lbl_dist}")

    if ranking_df["predicted_growth_2026"].isna().any():
        _fail("NaN predictions in investment_ranking")
    if pred_test_df["predicted_growth"].isna().any():
        _fail("NaN predictions in predictions_test")
    _ok("No NaN predictions in either output file")

    # ── git status ────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("GIT STATUS")
    print(SEP)
    result = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(result.stdout if result.stdout else "  (clean)")

    print(f"\n{SEP}")
    print("DONE — Phase V2.4 complete")
    print(SEP)


if __name__ == "__main__":
    main()
