"""
01_ml_prototype.py

scikit-learn ML prototype for one-year-ahead US housing price growth prediction.

Models trained:
  Naive baseline      — predict median training target for every row
  Model A  (Ridge)    — housing history only, linear
  Model A  (RF)       — housing history only, Random Forest
  Model C  (Ridge)    — housing history + weather/climate, linear
  Model C  (RF)       — housing history + weather/climate, Random Forest

Best model selected by: highest directional accuracy, tie-break lowest MAE.

Input:  Methoden Data/Modeling Data/Annual_Modeling_Table.csv
Outputs:
  app/data/metrics.csv
  app/data/predictions_test.csv
  app/data/investment_ranking.csv
  app/models/best_model.pkl
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT     = Path(__file__).resolve().parent.parent
IN_FILE  = ROOT / "Methoden Data" / "Modeling Data" / "Annual_Modeling_Table.csv"
OUT_DIR  = ROOT / "app" / "data"
MDL_DIR  = ROOT / "app" / "models"

METRICS_OUT    = OUT_DIR / "metrics.csv"
PRED_TEST_OUT  = OUT_DIR / "predictions_test.csv"
RANKING_OUT    = OUT_DIR / "investment_ranking.csv"
MODEL_OUT      = MDL_DIR / "best_model.pkl"

TARGET_COL = "target_growth_1y"

FEATURES_A = [
    "zhvi_sfr_mid_dec",
    "zhvi_sfr_mid_dec_prev",
    "growth_prev",
]

LABEL_THRESHOLDS = [
    ( 0.06,  float("inf"), "Attractive"),
    ( 0.02,  0.06,         "Moderate"),
    (-0.02,  0.02,         "Neutral"),
    (-float("inf"), -0.02, "Caution"),
]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def recommendation_label(growth: float) -> str:
    for lo, hi, label in LABEL_THRESHOLDS:
        if lo <= growth < hi:
            return label
    return "Caution"


def compute_metrics(y_true, y_pred, model_name: str, feature_set: str) -> dict:
    mae      = mean_absolute_error(y_true, y_pred) * 100
    rmse     = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    r2       = r2_score(y_true, y_pred)
    dir_acc  = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
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
        ("model",   RandomForestRegressor(n_estimators=200, random_state=42)),
    ])


def _fail(msg: str):
    print(f"\n[SANITY FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def _ok(msg: str):
    print(f"  [OK] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# pre-flight sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def pre_flight(df: pd.DataFrame, features_c: list[str]):
    print("\n── PRE-FLIGHT SANITY CHECKS ─────────────────────────────────────")

    # 1. input file existence confirmed by caller before load; echo shape
    _ok(f"Input loaded — shape {df.shape[0]:,} rows × {df.shape[1]} columns")

    # 2. year range and split counts
    yr_min, yr_max = df["YEAR"].min(), df["YEAR"].max()
    split_counts = df["split"].value_counts().to_dict()
    _ok(f"Year range: {yr_min}–{yr_max}")
    _ok(f"Split counts: {split_counts}")

    # 3. required columns
    required = [
        "CBSA_CODE", "CBSA_TITLE", "YEAR", "split", TARGET_COL,
        "zhvi_sfr_mid_dec", "zhvi_sfr_mid_dec_prev", "growth_prev",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        _fail(f"Missing required columns: {missing_cols}")
    _ok(f"All {len(required)} required columns present")

    # 4. train/test rows must have non-null target
    for sp in ("train", "test"):
        null_tgt = df.loc[df["split"] == sp, TARGET_COL].isna().sum()
        if null_tgt > 0:
            _fail(f"{null_tgt} rows in split='{sp}' have null {TARGET_COL}")
    _ok("train and test rows all have non-null target")

    # 5. predict rows may have null target (expected for 2025)
    predict_null = df.loc[df["split"] == "predict", TARGET_COL].isna().sum()
    predict_total = (df["split"] == "predict").sum()
    _ok(f"predict rows: {predict_total} total, {predict_null} with null target (expected)")

    # 6. target column not in either feature set
    for fs_name, fs in [("A", FEATURES_A), ("C-lite", features_c)]:
        if TARGET_COL in fs:
            _fail(f"Target column found in Feature set {fs_name}")
    _ok("Target column absent from both feature sets")

    # 7. no "next" columns in feature sets
    for fs_name, fs in [("A", FEATURES_A), ("C-lite", features_c)]:
        next_cols = [c for c in fs if "next" in c.lower()]
        if next_cols:
            _fail(f"Leakage: 'next' column(s) in Feature set {fs_name}: {next_cols}")
    _ok("No 'next' columns in any feature set")

    # 8. Feature set A has exactly 3 housing-history features
    if len(FEATURES_A) != 3:
        _fail(f"Feature set A must have 3 features, got {len(FEATURES_A)}")
    _ok(f"Feature set A has exactly 3 features: {FEATURES_A}")

    # 9. Feature set C-lite = A + all _ann columns
    ann_cols_in_c    = [c for c in features_c if c.endswith("_ann")]
    non_ann_in_c     = [c for c in features_c if not c.endswith("_ann") and c not in FEATURES_A]
    all_ann_in_table = sorted(c for c in df.columns if c.endswith("_ann"))
    missing_ann      = [c for c in all_ann_in_table if c not in features_c]
    if non_ann_in_c:
        _fail(f"Feature set C-lite contains unexpected non-_ann, non-A features: {non_ann_in_c}")
    if missing_ann:
        _fail(f"Feature set C-lite missing _ann columns: {missing_ann}")
    _ok(f"Feature set C-lite = Feature set A ({len(FEATURES_A)}) "
        f"+ {len(ann_cols_in_c)} _ann weather features = {len(features_c)} total")

    # 10. missing-value counts for both feature sets (train rows)
    train_df = df[df["split"] == "train"]
    print()
    print("  Missing values in train rows:")
    for fs_name, fs in [("A", FEATURES_A), ("C-lite", features_c)]:
        mv = train_df[fs].isna().sum()
        total_mv = mv.sum()
        print(f"    Feature set {fs_name}: {total_mv} total missing cells", end="")
        if total_mv > 0:
            print(f" — {mv[mv > 0].to_dict()}")
        else:
            print(" — none")

    # 11. target summary stats for train and test separately
    print()
    print("  Target summary statistics:")
    for sp in ("train", "test"):
        t = df.loc[df["split"] == sp, TARGET_COL] * 100
        print(f"    {sp:6s}: n={len(t):,}  mean={t.mean():+.2f}%  "
              f"std={t.std():.2f}%  min={t.min():+.2f}%  max={t.max():+.2f}%  "
              f"pct_pos={( t > 0).mean()*100:.1f}%")

    # 12. year ranges per split
    print()
    for sp in ("train", "test", "predict"):
        yrs = df.loc[df["split"] == sp, "YEAR"]
        print(f"  {sp:8s}: years {yrs.min()}–{yrs.max()}")

    if df.loc[df["split"] == "train", "YEAR"].max() > 2021:
        _fail("train set contains years after 2021")
    if not (df.loc[df["split"] == "test", "YEAR"].between(2022, 2024).all()):
        _fail("test set contains years outside 2022–2024")
    if not (df.loc[df["split"] == "predict", "YEAR"] == 2025).all():
        _fail("predict set contains years other than 2025")
    _ok("Year assignments correct: train ≤ 2021, test 2022–2024, predict = 2025")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# post-flight output checks
# ─────────────────────────────────────────────────────────────────────────────

def post_flight(metrics_df, pred_test_df, ranking_df, best_row, n_predict_rows):
    print("\n── POST-FLIGHT OUTPUT CHECKS ────────────────────────────────────")

    # 1. all four output files exist
    for path in (METRICS_OUT, PRED_TEST_OUT, RANKING_OUT, MODEL_OUT):
        if not path.exists():
            _fail(f"Output file missing: {path}")
        _ok(f"Exists: {path.name}")

    # 2. metrics table
    print()
    print("  Metrics table:")
    cols = ["model", "feature_set", "mae_pct", "rmse_pct", "r2", "directional_accuracy"]
    print(metrics_df[cols].to_string(index=False))

    # 3. best model
    print()
    print(f"  Best model : {best_row['model']} ({best_row['feature_set']})")
    print(f"  Reason     : directional_accuracy={best_row['directional_accuracy']:.4f}, "
          f"mae={best_row['mae_pct']:.4f} pct-pts")

    # 4. predictions_test shape and first rows
    print()
    print(f"  predictions_test.csv : {pred_test_df.shape[0]:,} rows × {pred_test_df.shape[1]} cols")
    print(pred_test_df.head(5).to_string(index=False))

    # 5. investment_ranking top 10
    print()
    print(f"  investment_ranking.csv : {ranking_df.shape[0]:,} rows × {ranking_df.shape[1]} cols")
    print("  Top 10 by predicted growth:")
    top10_cols = ["CBSA_CODE", "CBSA_TITLE", "predicted_growth_2026", "recommendation_label"]
    print(ranking_df.sort_values("predicted_growth_2026", ascending=False)
                    .head(10)[top10_cols].to_string(index=False))

    # 6. roughly one row per CBSA in predict split
    if ranking_df.shape[0] != n_predict_rows:
        _fail(f"investment_ranking has {ranking_df.shape[0]} rows but predict split has {n_predict_rows}")
    _ok(f"investment_ranking has {ranking_df.shape[0]} rows — matches predict split")

    # 7. no NaN predictions
    if ranking_df["predicted_growth_2026"].isna().any():
        _fail("investment_ranking contains NaN predictions")
    if pred_test_df["predicted_growth"].isna().any():
        _fail("predictions_test contains NaN predictions")
    _ok("No NaN predictions in either output file")

    # 8. recommendation labels populated and valid
    valid_labels = {"Attractive", "Moderate", "Neutral", "Caution"}
    bad = set(ranking_df["recommendation_label"].unique()) - valid_labels
    if bad:
        _fail(f"Unexpected recommendation labels: {bad}")
    label_counts = ranking_df["recommendation_label"].value_counts().to_dict()
    _ok(f"Recommendation labels valid — distribution: {label_counts}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── confirm input exists ──────────────────────────────────────────────────
    if not IN_FILE.exists():
        print(f"[ERROR] Input file not found: {IN_FILE}", file=sys.stderr)
        sys.exit(1)

    # ── load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(IN_FILE, dtype={"CBSA_CODE": str})

    # feature set C-lite derived from the actual table columns
    FEATURES_C = FEATURES_A + sorted(c for c in df.columns if c.endswith("_ann"))

    # ── pre-flight checks ─────────────────────────────────────────────────────
    pre_flight(df, FEATURES_C)

    # ── split ─────────────────────────────────────────────────────────────────
    train_df   = df[df["split"] == "train"].copy()
    test_df    = df[df["split"] == "test"].copy()
    predict_df = df[df["split"] == "predict"].copy()

    y_train = train_df[TARGET_COL].values
    y_test  = test_df[TARGET_COL].values

    # ── define models ─────────────────────────────────────────────────────────
    models = [
        ("Ridge",         "A",      build_ridge_pipeline(), FEATURES_A),
        ("RandomForest",  "A",      build_rf_pipeline(),    FEATURES_A),
        ("Ridge",         "C-lite", build_ridge_pipeline(), FEATURES_C),
        ("RandomForest",  "C-lite", build_rf_pipeline(),    FEATURES_C),
    ]

    print("── TRAINING ─────────────────────────────────────────────────────")

    all_metrics = []
    trained_models = {}

    # naive baseline
    naive_pred_value = float(np.median(y_train))
    naive_preds = np.full_like(y_test, fill_value=naive_pred_value, dtype=float)
    naive_m = compute_metrics(y_test, naive_preds, "Naive baseline", "—")
    all_metrics.append(naive_m)
    print(f"  Naive baseline trained  "
          f"(median={naive_pred_value*100:+.2f}%)  "
          f"dir_acc={naive_m['directional_accuracy']:.4f}  mae={naive_m['mae_pct']:.4f}")

    for model_name, fs_name, pipe, features in models:
        X_train = train_df[features].values
        X_test  = test_df[features].values
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = compute_metrics(y_test, preds, model_name, fs_name)
        all_metrics.append(m)
        trained_models[(model_name, fs_name)] = (pipe, features)
        print(f"  {model_name:15s} [{fs_name:6s}]  "
              f"dir_acc={m['directional_accuracy']:.4f}  "
              f"mae={m['mae_pct']:.4f}  rmse={m['rmse_pct']:.4f}  r2={m['r2']:.4f}")

    # ── select best model ─────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    # sort: directional_accuracy desc, mae_pct asc; skip naive row for selection
    candidate_metrics = metrics_df[metrics_df["model"] != "Naive baseline"].copy()
    candidate_metrics = candidate_metrics.sort_values(
        ["directional_accuracy", "mae_pct"],
        ascending=[False, True],
    ).reset_index(drop=True)
    best_row     = candidate_metrics.iloc[0]
    best_key     = (best_row["model"], best_row["feature_set"])
    best_pipe, best_features = trained_models[best_key]

    print(f"\n  Selected best: {best_row['model']} [{best_row['feature_set']}]  "
          f"(dir_acc={best_row['directional_accuracy']:.4f}, "
          f"mae={best_row['mae_pct']:.4f} pct-pts)")

    # ── save metrics ─────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_OUT, index=False)

    # ── predictions_test.csv ─────────────────────────────────────────────────
    X_test_best = test_df[best_features].values
    preds_test  = best_pipe.predict(X_test_best)

    pred_test_df = pd.DataFrame({
        "CBSA_CODE":        test_df["CBSA_CODE"].values,
        "CBSA_TITLE":       test_df["CBSA_TITLE"].values,
        "YEAR":             test_df["YEAR"].values,
        "actual_growth":    np.round(y_test * 100, 4),
        "predicted_growth": np.round(preds_test * 100, 4),
        "abs_error":        np.round(np.abs(y_test - preds_test) * 100, 4),
        "direction_correct": (np.sign(preds_test) == np.sign(y_test)).astype(int),
    })
    pred_test_df.to_csv(PRED_TEST_OUT, index=False)

    # ── investment_ranking.csv ────────────────────────────────────────────────
    X_predict = predict_df[best_features].values
    preds_2026 = best_pipe.predict(X_predict)

    ranking_df = pd.DataFrame({
        "CBSA_CODE":             predict_df["CBSA_CODE"].values,
        "CBSA_TITLE":            predict_df["CBSA_TITLE"].values,
        "YEAR":                  predict_df["YEAR"].values,
        "predicted_growth_2026": np.round(preds_2026 * 100, 4),
        "recommendation_label":  [recommendation_label(g) for g in preds_2026],
        "zhvi_sfr_mid_dec":      predict_df["zhvi_sfr_mid_dec"].values,
    })

    # optional columns — include if present
    for optional_col in ("had_major_disaster_ann", "damage_property_sum_ann"):
        if optional_col in predict_df.columns:
            ranking_df[optional_col] = predict_df[optional_col].values

    ranking_df = ranking_df.sort_values("predicted_growth_2026", ascending=False).reset_index(drop=True)
    ranking_df.to_csv(RANKING_OUT, index=False)

    # ── save best model ───────────────────────────────────────────────────────
    MDL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "features": best_features, "name": best_key}, MODEL_OUT)

    # ── post-flight checks ────────────────────────────────────────────────────
    post_flight(metrics_df, pred_test_df, ranking_df, best_row, n_predict_rows=len(predict_df))

    print("── DONE ─────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
