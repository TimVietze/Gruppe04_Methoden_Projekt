"""
absolute_modeling_table_adjustment.py

Build the LONG-format modeling table for the 6-month absolute-price model.
Each (CBSA, month) is expanded into up to 9 rows — one per housing category —
so the model has:

    price_now, price_next_6m, cat_<category> dummies (9), and all the
    weather / disaster / economic features unchanged.

Compared to the 9-target wide format this:
  - lets one model learn shared dynamics across categories
  - drops only individual (CBSA, month, category) rows when a single category
    is missing, instead of losing the whole CBSA-month
  - keeps standard single-output R² / RMSE scoring

The 9 wide `zhvi_*` (current) and `zhvi_*_next_6m` (future) columns are NOT
kept — they're replaced by `price_now` / `price_next_6m` for the row's
category. `zhvi_avg_next_6m` and `price_change_next_6m` are also dropped.

Input:  Methoden Data/Modeling_Table.csv      (canonical, untouched)
Output: ./Modeling_Table_absolute.csv         (overwritten)
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# .../1 Table Adjustment → .../Prediction... → .../Model Analysis → project root
ROOT = HERE.parent.parent.parent
SOURCE = ROOT / "Methoden Data" / "Modeling_Table.csv"
TARGET = HERE / "Modeling_Table_absolute.csv"

# Short category labels used in the cat_<...> dummies.
CATEGORIES = [
    "all_bottom", "all_top",
    "sfr_mid",    "condo_mid",
    "1br_mid", "2br_mid", "3br_mid", "4br_mid", "5br_mid",
]
PRICE_NOW_COLS  = [f"zhvi_{c}"         for c in CATEGORIES]
PRICE_NEXT_COLS = [f"zhvi_{c}_next_6m" for c in CATEGORIES]
EXTRA_DROP = ["zhvi_avg_next_6m", "price_change_next_6m"]

# Restrict the modeling table to the post-2015 period to keep the file size manageable.
START_DATE = "2015-01"


def main() -> None:
    df = pd.read_csv(SOURCE)

    missing = [c for c in PRICE_NOW_COLS + PRICE_NEXT_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Source is missing expected columns: {missing}")

    df = df[df["YEAR_MONTH"] >= START_DATE].reset_index(drop=True)

    id_cols = [c for c in df.columns
               if c not in PRICE_NOW_COLS + PRICE_NEXT_COLS + EXTRA_DROP]

    parts = []
    for cat, now_col, next_col in zip(CATEGORIES, PRICE_NOW_COLS, PRICE_NEXT_COLS):
        part = df[id_cols].copy()
        part["category"] = cat
        part["price_now"] = df[now_col]
        part["price_next_6m"] = df[next_col]
        parts.append(part)

    long_df = pd.concat(parts, ignore_index=True)

    # A category-row with no current or future price carries no signal.
    long_df = long_df.dropna(subset=["price_now", "price_next_6m"]).reset_index(drop=True)

    dummies = pd.get_dummies(long_df["category"], prefix="cat", dtype=int)
    long_df = pd.concat([long_df.drop(columns=["category"]), dummies], axis=1)

    # Log-transformed price columns. Linear models predict additive effects;
    # in $-space, category and metro effects compound multiplicatively (a
    # 5br in Manhattan ≠ Manhattan_mean + 5br_mean − global_mean). Training
    # on log(price) turns those multiplicative effects into additive ones,
    # which is exactly what linear models can fit. The non-log columns are
    # kept alongside for inspection / evaluation in $.
    long_df["log_price_now"] = np.log(long_df["price_now"])
    long_df["log_price_next_6m"] = np.log(long_df["price_next_6m"])

    final_cols = (
        id_cols
        + list(dummies.columns)
        + ["price_now", "log_price_now", "price_next_6m", "log_price_next_6m"]
    )
    long_df = long_df[final_cols]

    long_df.to_csv(TARGET, index=False)
    print(
        f"Wrote {TARGET.name}: {len(long_df):,} rows × {long_df.shape[1]} cols "
        f"(from {len(df):,} wide rows, {len(CATEGORIES)} categories)"
    )


if __name__ == "__main__":
    main()
