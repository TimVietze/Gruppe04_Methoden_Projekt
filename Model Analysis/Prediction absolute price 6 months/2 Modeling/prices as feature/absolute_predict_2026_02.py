"""absolute_predict_2026_02.py

Inference-only script. Loads the trained ridge model and predicts absolute
prices 6 months ahead from the YEAR_MONTH = 2026-02 baseline (target = Aug 2026).

Pipeline matches the training script exactly so the saved Pipeline.predict() works:
  - reshape wide → long (one row per CBSA × category)
  - forward-fill `unemployment_rate_monthly` from the last known value per CBSA
    (BLS publishes the series with a 2-month lag — Dec 2025 is the latest)
  - add month / quarter / year time features
  - build the same feature DataFrame, in the same column order, as training

The predictions are appended to:
  3 Results/prices as feature/results prices/absolute_predictions_ridge.csv
with NaN in the `actual_*` columns (since the 6m-ahead target is unknown).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent.parent
SOURCE = ROOT / "Methoden Data" / "Modeling_Table.csv"
MODEL = (
    HERE.parent.parent / "3 Results" / "prices as feature" / "absolute_ridge.pkl"
)
FEATURE_LIST = (
    HERE.parent.parent
    / "3 Results"
    / "prices as feature"
    / "absolute_feature_columns.txt"
)
PRED_CSV = (
    HERE.parent.parent
    / "3 Results"
    / "prices as feature"
    / "results prices"
    / "absolute_predictions_ridge.csv"
)

BASELINE_YM = "2026-02"
TARGET_YM_STR = "2026-08"  # informational; goes into a docstring only

CATEGORIES = [
    "all_bottom", "all_top",
    "sfr_mid",    "condo_mid",
    "1br_mid", "2br_mid", "3br_mid", "4br_mid", "5br_mid",
]
PRICE_NOW_COLS = [f"zhvi_{c}" for c in CATEGORIES]


def build_long_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide-to-long expansion that matches the adjustment script, but keeps
    rows whose 6m-ahead target is missing (we're forecasting that)."""
    drop_cols = PRICE_NOW_COLS + [f"{c}_next_6m" for c in PRICE_NOW_COLS] + [
        "zhvi_avg_next_6m", "price_change_next_6m"
    ]
    id_cols = [c for c in df_wide.columns if c not in drop_cols]
    parts = []
    for cat, now_col in zip(CATEGORIES, PRICE_NOW_COLS):
        part = df_wide[id_cols].copy()
        part["category"] = cat
        part["price_now"] = df_wide[now_col]
        parts.append(part)
    long_df = pd.concat(parts, ignore_index=True)
    long_df = long_df.dropna(subset=["price_now"]).reset_index(drop=True)
    dummies = pd.get_dummies(long_df["category"], prefix="cat", dtype=int)
    long_df = pd.concat([long_df.drop(columns=["category"]), dummies], axis=1)
    long_df["log_price_now"] = np.log(long_df["price_now"])
    long_df["YEAR_MONTH"] = pd.to_datetime(long_df["YEAR_MONTH"])
    long_df["month"] = long_df["YEAR_MONTH"].dt.month
    long_df["quarter"] = long_df["YEAR_MONTH"].dt.quarter
    long_df["year"] = long_df["YEAR_MONTH"].dt.year
    # Also keep the original category label for the output (was dropped into dummies)
    long_df["category"] = long_df[[c for c in long_df.columns if c.startswith("cat_")]].idxmax(axis=1).str.replace("cat_", "", regex=False)
    return long_df


def main() -> None:
    print(f"Loading {SOURCE.name} ...")
    raw = pd.read_csv(SOURCE)

    # Forward-fill unemployment per CBSA so the Feb 2026 rows have a value.
    print("Forward-filling unemployment_rate_monthly per CBSA ...")
    raw = raw.sort_values(["CBSA_CODE", "YEAR_MONTH"]).reset_index(drop=True)
    raw["unemployment_rate_monthly"] = raw.groupby("CBSA_CODE")["unemployment_rate_monthly"].ffill()
    n_still_nan = raw.loc[raw["YEAR_MONTH"] == BASELINE_YM, "unemployment_rate_monthly"].isna().sum()
    if n_still_nan:
        print(f"  WARNING: {n_still_nan} Feb-2026 rows still have NaN unemployment after ffill.")

    feb = raw[raw["YEAR_MONTH"] == BASELINE_YM].copy()
    print(f"Feb 2026 base rows: {len(feb)} ({feb['CBSA_CODE'].nunique()} CBSAs)")

    long_df = build_long_features(feb)
    print(f"Long-format inference rows: {len(long_df)} (price_now present)")

    feature_cols = FEATURE_LIST.read_text().splitlines()
    feature_cols = [c for c in feature_cols if c.strip()]
    print(f"Model expects {len(feature_cols)} features.")

    missing = [c for c in feature_cols if c not in long_df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns in inference frame: {missing}")

    # Drop any inference row where a non-imputable feature is still NaN.
    before = len(long_df)
    long_df = long_df.dropna(subset=feature_cols).reset_index(drop=True)
    after = len(long_df)
    print(f"Dropped {before - after} rows with NaN in any model feature; {after} remain.")

    X = long_df[feature_cols]

    print(f"Loading {MODEL.name} ...")
    model = joblib.load(MODEL)
    print("Predicting log prices ...")
    y_log = model.predict(X)
    y_abs = np.exp(y_log)

    pred = pd.DataFrame({
        "CBSA_CODE":                   long_df["CBSA_CODE"].astype(int),
        "CBSA_TITLE":                  long_df["CBSA_TITLE"],
        "YEAR_MONTH":                  long_df["YEAR_MONTH"].dt.strftime("%Y-%m-%d"),
        "category":                    long_df["category"],
        "price_now":                   long_df["price_now"],
        "actual_price_next_6m":        np.nan,  # unknown — true forecast
        "predicted_price_next_6m":     y_abs,
        "actual_log_price_next_6m":    np.nan,
        "predicted_log_price_next_6m": y_log,
    })

    print(f"Inference rows produced: {len(pred)} "
          f"(target horizon = {TARGET_YM_STR})")

    print(f"Appending to {PRED_CSV.name} ...")
    existing = pd.read_csv(PRED_CSV)
    # Idempotency: remove any prior 2026-02 rows so re-runs don't duplicate.
    n_existing_baseline = (existing["YEAR_MONTH"] == BASELINE_YM + "-01").sum()
    if n_existing_baseline:
        print(f"  Removing {n_existing_baseline} existing {BASELINE_YM} rows before append.")
        existing = existing[existing["YEAR_MONTH"] != BASELINE_YM + "-01"]
    combined = pd.concat([existing, pred[existing.columns.tolist()]], ignore_index=True)
    combined.to_csv(PRED_CSV, index=False)
    print(f"Wrote {len(combined)} total rows to {PRED_CSV.name}.")


if __name__ == "__main__":
    main()
