"""
absolute_modeling_training_testing_no_prices.py

Same setup as ../prices as feature/absolute_modeling_training_testing.py
BUT with BOTH `price_now` and `log_price_now` removed from the feature set.
This isolates how much signal the remaining features carry on their own
— i.e. how well we can predict the absolute price 6 months out *without*
using today's price as an anchor.

The CBSA identity IS included as a feature (one-hot for linear models,
ordinal for tree models). This is essential once the price anchor is gone,
because otherwise the model has no way to tell which metro it's looking at.

Target: log(price_next_6m) — same log-$ training / exp-back-to-$ reporting
as the sister script. Reads the long-format Modeling_Table_absolute.csv
directly (no new CSV is exported).

Models compared (5): linear_regression, ridge, random_forest, xgboost,
                     gradient_boosting (HistGradientBoostingRegressor)

Input:  ../../1 Table Adjustment/Modeling_Table_absolute.csv
Output: ../../3 Results/no prices as feature/  (prefix 'absolute_no_prices_')
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                                   # .../2 Modeling/no prices as feature
DATA_FILE = HERE.parent.parent / "1 Table Adjustment" / "Modeling_Table_absolute.csv"
OUT_DIR = HERE.parent.parent / "3 Results" / "no prices as feature"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_PREDS = OUT_DIR / "results no prices"
OUT_DIR_PREDS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------
TARGET = "log_price_next_6m"           # train on log $
TARGET_ABS = "price_next_6m"           # absolute $, used only for evaluation
META_COLS = ["CBSA_TITLE", "YEAR_MONTH"]
# Drop every price-related column from features so the model has no anchor
# on today's level. price_next_6m would also leak the target after exp().
EXCLUDE_FROM_FEATURES = ["price_now", "log_price_now", TARGET_ABS]
CBSA_COL = "CBSA_CODE"
CAT_COLS = [
    "cat_1br_mid", "cat_2br_mid", "cat_3br_mid", "cat_4br_mid", "cat_5br_mid",
    "cat_all_bottom", "cat_all_top", "cat_condo_mid", "cat_sfr_mid",
]

# ----------------------------------------------------------------------------
# Search settings
# ----------------------------------------------------------------------------
N_ITER = 10
CV_FOLDS = 5
RANDOM_STATE = 42
HOLDOUT_MONTHS = 12

MODEL_PREFIX = "absolute_no_prices_"


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
    df["month"] = df["YEAR_MONTH"].dt.month
    df["quarter"] = df["YEAR_MONTH"].dt.quarter
    df["year"] = df["YEAR_MONTH"].dt.year
    df = df.sort_values("YEAR_MONTH").reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in META_COLS + [TARGET] + EXCLUDE_FROM_FEATURES]
    df = df.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Preprocessor builders
# ----------------------------------------------------------------------------
def linear_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num",  StandardScaler(), numeric_cols),
        ("cbsa", OneHotEncoder(handle_unknown="ignore", sparse_output=True), [CBSA_COL]),
    ])


def tree_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num",  "passthrough", numeric_cols),
        ("cbsa", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                 [CBSA_COL]),
    ])


# ----------------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------------
def model_space(numeric_cols: list[str]) -> dict:
    pipe_lin = Pipeline([
        ("pre", linear_preprocessor(numeric_cols)),
        ("est", LinearRegression()),
    ])
    pipe_ridge = Pipeline([
        ("pre", linear_preprocessor(numeric_cols)),
        ("est", Ridge(random_state=RANDOM_STATE)),
    ])
    pipe_rf = Pipeline([
        ("pre", tree_preprocessor(numeric_cols)),
        ("est", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
    ])
    pipe_xgb = Pipeline([
        ("pre", tree_preprocessor(numeric_cols)),
        ("est", xgb.XGBRegressor(
            random_state=RANDOM_STATE, n_jobs=1, verbosity=0, tree_method="hist",
        )),
    ])
    pipe_histgb = Pipeline([
        ("pre", tree_preprocessor(numeric_cols)),
        # CBSA stays ordinal-numeric here (sklearn's HistGB caps native categorical
        # features at 255 cardinality; we have 855 CBSAs).
        ("est", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])

    return {
        "linear_regression": (pipe_lin, {}),
        "ridge": (pipe_ridge, {"est__alpha": loguniform(1e-3, 1e3)}),
        "random_forest": (
            pipe_rf,
            {
                "est__n_estimators":      randint(100, 250),
                "est__max_depth":         randint(6, 18),
                "est__min_samples_split": randint(5, 30),
                "est__min_samples_leaf":  randint(5, 30),
                "est__max_features":      ["sqrt", "log2"],
            },
        ),
        "xgboost": (
            pipe_xgb,
            {
                "est__n_estimators":     randint(150, 800),
                "est__max_depth":        randint(3, 12),
                "est__learning_rate":    loguniform(0.01, 0.3),
                "est__subsample":        uniform(0.6, 0.4),
                "est__colsample_bytree": uniform(0.6, 0.4),
                "est__min_child_weight": randint(1, 30),
                "est__reg_alpha":        loguniform(1e-3, 10),
                "est__reg_lambda":       loguniform(1e-3, 10),
            },
        ),
        "gradient_boosting": (
            pipe_histgb,
            {
                "est__max_iter":          randint(150, 500),
                "est__max_depth":         randint(3, 12),
                "est__learning_rate":     loguniform(0.01, 0.3),
                "est__min_samples_leaf":  randint(20, 200),
                "est__max_leaf_nodes":    randint(15, 127),
                "est__l2_regularization": loguniform(1e-3, 10),
            },
        ),
    }


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def per_category_eval(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for cat_col in CAT_COLS:
        mask = df_test[cat_col].to_numpy() == 1
        if mask.sum() == 0:
            continue
        rows.append({
            "Category": cat_col.replace("cat_", ""),
            "N_Rows":   int(mask.sum()),
            **evaluate(y_true[mask], y_pred[mask]),
        })
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run() -> None:
    print("=" * 80)
    print("ABSOLUTE PRICE 6M  —  TRAIN / TEST (NO price_now, WITH CBSA metro identity)")
    print("=" * 80)

    df = load_data()
    print(f"Clean rows: {len(df):,}  "
          f"({df['YEAR_MONTH'].min().date()} → {df['YEAR_MONTH'].max().date()})")

    cutoff = df["YEAR_MONTH"].max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    train = df[df["YEAR_MONTH"] <= cutoff]
    test  = df[df["YEAR_MONTH"] >  cutoff]

    feature_cols = [c for c in df.columns
                    if c not in META_COLS + [TARGET] + EXCLUDE_FROM_FEATURES]
    numeric_cols = [c for c in feature_cols if c != CBSA_COL]
    n_cbsa = train[CBSA_COL].nunique()

    X_train = train[feature_cols]
    y_train = train[TARGET].to_numpy()       # log $
    X_test  = test[feature_cols]
    y_test_abs = test[TARGET_ABS].to_numpy() # absolute $ for evaluation

    print(f"Train: {len(train):,}   Test: {len(test):,}   Split @ {cutoff.date()}")
    print(f"Features: {len(numeric_cols)} numeric + 1 CBSA ({n_cbsa:,} unique codes)")
    print(f"Excluded from features: {EXCLUDE_FROM_FEATURES}")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    space = model_space(numeric_cols)

    # Pre-compute category label per test row so each per-model prediction CSV
    # can join it back without recomputing.
    cat_labels = np.array([c.replace("cat_", "") for c in CAT_COLS])
    test_categories = cat_labels[test[CAT_COLS].to_numpy().argmax(axis=1)]

    comparison_rows: list[dict] = []
    per_cat_rows: list[dict] = []
    best_params: dict[str, dict] = {}
    cv_scores_log: dict[str, float] = {}

    for name, (estimator, grid) in space.items():
        print(f"\n--- {name} ---")
        if grid:
            search = RandomizedSearchCV(
                estimator,
                param_distributions=grid,
                n_iter=N_ITER,
                cv=tscv,
                scoring="neg_root_mean_squared_error",  # log space
                n_jobs=-1,
                random_state=RANDOM_STATE,
                refit=True,
                verbose=1,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params[name] = search.best_params_
            cv_scores_log[name] = float(-search.best_score_)
            print(f"  Best CV RMSE (log $): {cv_scores_log[name]:.4f}")
        else:
            best = estimator.fit(X_train, y_train)
            best_params[name] = {}
            cv_scores_log[name] = float("nan")

        y_pred_log = best.predict(X_test)
        y_pred_abs = np.exp(y_pred_log)
        agg = evaluate(y_test_abs, y_pred_abs)
        comparison_rows.append({
            "Model": name,
            "CV_RMSE_log": cv_scores_log[name],
            **agg,
        })
        for row in per_category_eval(test, y_test_abs, y_pred_abs):
            row["Model"] = name
            per_cat_rows.append(row)
        print(f"  Test  RMSE: ${agg['RMSE']:,.2f}   MAE: ${agg['MAE']:,.2f}   R²: {agg['R2']:.4f}")

        out_path = OUT_DIR / f"{MODEL_PREFIX}{name}.pkl"
        joblib.dump(best, out_path)
        print(f"  Saved → {out_path.name}")

        preds_df = test[["CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH"]].copy().reset_index(drop=True)
        preds_df["category"] = test_categories
        # price_now is kept for reference only — it was NOT a feature for this model.
        preds_df["price_now"] = test["price_now"].to_numpy()
        preds_df["actual_price_next_6m"] = y_test_abs
        preds_df["predicted_price_next_6m"] = y_pred_abs
        preds_df["actual_log_price_next_6m"] = test[TARGET].to_numpy()
        preds_df["predicted_log_price_next_6m"] = y_pred_log
        preds_path = OUT_DIR_PREDS / f"{MODEL_PREFIX}predictions_{name}.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"  Saved → {preds_path.relative_to(OUT_DIR)}")

    # ----- write comparison artifacts -----
    comp_df = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)
    comp_df.to_csv(OUT_DIR / f"{MODEL_PREFIX}01_Model_Comparison.csv", index=False)

    pcat_df = pd.DataFrame(per_cat_rows)[["Model", "Category", "N_Rows", "RMSE", "MAE", "R2"]]
    pcat_df.to_csv(OUT_DIR / f"{MODEL_PREFIX}02_Per_Category_Results.csv", index=False)

    hp_df = pd.DataFrame([{"Model": k, "Params": str(v)} for k, v in best_params.items()])
    hp_df.to_csv(OUT_DIR / f"{MODEL_PREFIX}03_Best_Hyperparameters.csv", index=False)

    (OUT_DIR / f"{MODEL_PREFIX}feature_columns.txt").write_text("\n".join(feature_cols))

    # ----- best-model test predictions -----
    best_name = comp_df.iloc[0]["Model"]
    best_model = joblib.load(OUT_DIR / f"{MODEL_PREFIX}{best_name}.pkl")
    y_best_log = best_model.predict(X_test)
    y_best_abs = np.exp(y_best_log)

    pred_df = test[["CBSA_CODE", "CBSA_TITLE", "YEAR_MONTH"]].copy().reset_index(drop=True)
    pred_df["price_now"] = test["price_now"].to_numpy()         # kept for reference, NOT a feature
    pred_df["actual_price_next_6m"]    = y_test_abs                  # $
    pred_df["predicted_price_next_6m"] = y_best_abs                  # $
    pred_df["actual_log_price_next_6m"]    = test[TARGET].to_numpy() # log $
    pred_df["predicted_log_price_next_6m"] = y_best_log              # log $
    cat_labels = np.array([c.replace("cat_", "") for c in CAT_COLS])
    pred_df["category"] = cat_labels[test[CAT_COLS].to_numpy().argmax(axis=1)]
    pred_df.to_csv(OUT_DIR / f"{MODEL_PREFIX}04_Test_Predictions_Best.csv", index=False)

    # ----- summary report -----
    report = (
        f"ABSOLUTE PRICE 6M — TRAIN / TEST (log target, CBSA, NO price_now)\n"
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        f"Data:     {DATA_FILE.name}  ({len(df):,} clean long-format rows)\n"
        f"Target:   {TARGET}   (predictions exp()-converted back to $ for metrics)\n"
        f"Features: {len(numeric_cols)} numeric + 1 CBSA ({n_cbsa:,} unique codes)\n"
        f"          (price_now + log_price_now EXCLUDED — weather + econ + disaster + category dummies + time + CBSA)\n"
        f"CV:       TimeSeriesSplit({CV_FOLDS})   RandomizedSearch n_iter={N_ITER}\n"
        f"Split:    train ≤ {cutoff.date()}   /   test > {cutoff.date()}\n\n"
        f"MODEL RANKING (test set, $-scale RMSE / MAE / R²; CV_RMSE_log is log-$ CV score):\n"
        f"{comp_df.to_string(index=False)}\n\n"
        f"PER-CATEGORY (best model = {best_name}, $-scale):\n"
        f"{pcat_df[pcat_df['Model'] == best_name].to_string(index=False)}\n\n"
        f"Best model: {best_name}\n"
    )
    (OUT_DIR / f"{MODEL_PREFIX}00_Summary_Report.txt").write_text(report)
    print("\n" + report)
    print("DONE.")


if __name__ == "__main__":
    run()
