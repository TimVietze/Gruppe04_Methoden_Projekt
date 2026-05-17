"""
generate_results_graphs.py

Build comparison visuals for the 6-month absolute-price prediction results:

  1. 01_model_comparison.png       — RMSE / MAE / R² bar charts comparing
                                     all 5 models in both feature setups
                                     (with vs without price_now).
  2. 02_actual_vs_predicted.png    — Scatter of predicted vs actual price
                                     for the best model in each setup,
                                     coloured by housing category, with
                                     the y=x reference diagonal.
  3. 03_time_series.png            — Time-series view (2×2 grid) for four
                                     example metros: actual price 6 months
                                     out vs both models' predictions
                                     (with / without price_now) over the
                                     hold-out window.

Inputs:
  ../prices as feature/absolute_01_Model_Comparison.csv
  ../prices as feature/absolute_04_Test_Predictions_Best.csv
  ../no prices as feature/absolute_no_prices_01_Model_Comparison.csv
  ../no prices as feature/absolute_no_prices_04_Test_Predictions_Best.csv

Outputs are written into this folder.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent                          # .../3 Results/graphs
RESULTS_DIR = HERE.parent
PRICES_DIR = RESULTS_DIR / "prices as feature"
NO_PRICES_DIR = RESULTS_DIR / "no prices as feature"

MODEL_ORDER = ["linear_regression", "ridge", "random_forest", "xgboost", "gradient_boosting"]
COLOR_WITH = "#2b8cbe"      # blue   — with price_now
COLOR_WITHOUT = "#e6550d"   # orange — without price_now


# ----------------------------------------------------------------------------
# Figure 1 — model comparison (RMSE / MAE / R²)
# ----------------------------------------------------------------------------
def model_comparison_chart() -> None:
    with_df    = pd.read_csv(PRICES_DIR    / "absolute_01_Model_Comparison.csv")
    without_df = pd.read_csv(NO_PRICES_DIR / "absolute_no_prices_01_Model_Comparison.csv")

    def vals(df: pd.DataFrame, metric: str) -> list[float]:
        return [float(df[df["Model"] == m][metric].iloc[0]) for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [("RMSE", "RMSE ($)", True),
               ("MAE",  "MAE ($)",  False),
               ("R2",   "R²",       False)]

    x = np.arange(len(MODEL_ORDER))
    width = 0.38

    for ax, (metric, ylabel, log) in zip(axes, metrics):
        with_vals    = vals(with_df, metric)
        without_vals = vals(without_df, metric)

        ax.bar(x - width / 2, with_vals,    width, label="with price_now",    color=COLOR_WITH)
        ax.bar(x + width / 2, without_vals, width, label="without price_now", color=COLOR_WITHOUT)

        # numeric labels on each bar
        for xi, v in zip(x - width / 2, with_vals):
            ax.annotate(f"{v:,.2f}" if metric == "R2" else f"{v:,.0f}",
                        (xi, v), ha="center", va="bottom", fontsize=8)
        for xi, v in zip(x + width / 2, without_vals):
            ax.annotate(f"{v:,.2f}" if metric == "R2" else f"{v:,.0f}",
                        (xi, v), ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        if log:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Model comparison — 6-month absolute price prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "01_model_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ----------------------------------------------------------------------------
# Figure 2 — actual vs predicted scatter (best model in each setup)
# ----------------------------------------------------------------------------
def actual_vs_predicted_chart() -> None:
    pred_with    = pd.read_csv(PRICES_DIR    / "absolute_04_Test_Predictions_Best.csv")
    pred_without = pd.read_csv(NO_PRICES_DIR / "absolute_no_prices_04_Test_Predictions_Best.csv")

    def best_name_from_summary(folder: Path, prefix: str) -> str:
        """Pull the winner's name out of the per-folder Model_Comparison.csv."""
        comp = pd.read_csv(folder / f"{prefix}01_Model_Comparison.csv")
        return comp.sort_values("RMSE").iloc[0]["Model"]

    title_with    = f"With price_now (best: {best_name_from_summary(PRICES_DIR,    'absolute_')})"
    title_without = f"Without price_now (best: {best_name_from_summary(NO_PRICES_DIR, 'absolute_no_prices_')})"

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.colormaps["tab10"]

    for ax, df, title in zip(axes, [pred_with, pred_without], [title_with, title_without]):
        categories = sorted(df["category"].unique())
        for i, cat in enumerate(categories):
            sub = df[df["category"] == cat]
            ax.scatter(sub["actual_price_next_6m"],
                       sub["predicted_price_next_6m"],
                       s=8, alpha=0.18, color=cmap(i % 10), label=cat,
                       edgecolors="none")

        # y = x reference
        lo = float(min(df["actual_price_next_6m"].min(),
                       df["predicted_price_next_6m"].min()))
        hi = float(max(df["actual_price_next_6m"].max(),
                       df["predicted_price_next_6m"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6, label="perfect (y = x)")

        # metrics annotation
        residuals = df["actual_price_next_6m"] - df["predicted_price_next_6m"]
        rmse = float(np.sqrt((residuals ** 2).mean()))
        mae  = float(residuals.abs().mean())
        ss_res = float((residuals ** 2).sum())
        ss_tot = float(((df["actual_price_next_6m"] - df["actual_price_next_6m"].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(0.02, 0.98,
                f"RMSE: ${rmse:,.0f}\nMAE:   ${mae:,.0f}\nR²:    {r2:.4f}\nn = {len(df):,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                          edgecolor="grey"))

        ax.set_xlabel("Actual price in 6 months ($)")
        ax.set_ylabel("Predicted price in 6 months ($)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        leg = ax.legend(loc="lower right", fontsize=8, markerscale=2.5,
                        framealpha=0.85, title="category")
        for handle in leg.legend_handles:
            try:
                handle.set_alpha(1.0)
            except AttributeError:
                pass

    fig.suptitle("Actual vs predicted housing prices in 6 months",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = HERE / "02_actual_vs_predicted.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ----------------------------------------------------------------------------
# Figure 3 — time-series view for selected metros
# ----------------------------------------------------------------------------
TIME_SERIES_CATEGORY = "sfr_mid"
# Use distinctive substrings — "Miami" alone matches "Miami, OK" first alphabetically.
TIME_SERIES_METROS = ["New York", "Los Angeles", "Chicago", "Miami-Fort Lauderdale"]


def time_series_chart() -> None:
    pred_with    = pd.read_csv(PRICES_DIR    / "absolute_04_Test_Predictions_Best.csv")
    pred_without = pd.read_csv(NO_PRICES_DIR / "absolute_no_prices_04_Test_Predictions_Best.csv")
    for df in (pred_with, pred_without):
        df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"])
        # x-axis = the month the predicted/actual price refers to (= as-of + 6 months)
        df["target_month"] = df["YEAR_MONTH"] + pd.DateOffset(months=6)

    # Source table — needed to compute the per-metro historical (training-window)
    # mean price, which is essentially what the no-prices linear model reverts to.
    src_path = HERE.parent.parent / "1 Table Adjustment" / "Modeling_Table_absolute.csv"
    src = pd.read_csv(src_path)
    src["YEAR_MONTH"] = pd.to_datetime(src["YEAR_MONTH"])
    train_cutoff = src["YEAR_MONTH"].max() - pd.DateOffset(months=12)
    src_train_cat = src[(src["YEAR_MONTH"] <= train_cutoff) &
                        (src[f"cat_{TIME_SERIES_CATEGORY}"] == 1)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for ax, metro_substr in zip(axes, TIME_SERIES_METROS):
        w = pred_with[
            pred_with["CBSA_TITLE"].str.contains(metro_substr, regex=False) &
            (pred_with["category"] == TIME_SERIES_CATEGORY)
        ]
        wo = pred_without[
            pred_without["CBSA_TITLE"].str.contains(metro_substr, regex=False) &
            (pred_without["category"] == TIME_SERIES_CATEGORY)
        ]
        if w.empty:
            ax.set_title(f"{metro_substr} — no test rows for {TIME_SERIES_CATEGORY}")
            ax.axis("off")
            continue

        # Pin to a single CBSA in case the substring matches more than one
        cbsa_pick = sorted(w["CBSA_TITLE"].unique())[0]
        w  = w[w["CBSA_TITLE"]  == cbsa_pick].sort_values("target_month")
        wo = wo[wo["CBSA_TITLE"] == cbsa_pick].sort_values("target_month")

        ax.plot(w["target_month"], w["actual_price_next_6m"],
                color="black", marker="o", linewidth=2.0, label="actual")
        ax.plot(w["target_month"], w["predicted_price_next_6m"],
                color=COLOR_WITH, marker="s", linestyle="--", linewidth=1.6,
                label="pred (with price_now)")
        ax.plot(wo["target_month"], wo["predicted_price_next_6m"],
                color=COLOR_WITHOUT, marker="^", linestyle=":", linewidth=1.6,
                label="pred (without price_now)")

        # Historical mean of price_next_6m for this CBSA × sfr_mid over the
        # training window (2015 → 2024-08). The no-prices model effectively
        # reverts to this level.
        hist_mean = src_train_cat.loc[
            src_train_cat["CBSA_TITLE"] == cbsa_pick, "price_next_6m"
        ].mean()
        if pd.notna(hist_mean):
            ax.axhline(hist_mean, color="gray", linestyle="-.", linewidth=1.4, alpha=0.75,
                       label=f"2015–24 mean (${hist_mean:,.0f})")

        ax.set_title(cbsa_pick, fontsize=10)
        ax.set_ylabel("price ($)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(loc="best", fontsize=8)

    for ax in axes[-2:]:
        ax.set_xlabel("target month (= as-of + 6 months)")

    fig.suptitle(f"Time-series view: actual vs predicted prices ({TIME_SERIES_CATEGORY}, test set)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = HERE / "03_time_series.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def main() -> None:
    model_comparison_chart()
    actual_vs_predicted_chart()
    time_series_chart()
    print(f"All graphs written to: {HERE}")


if __name__ == "__main__":
    main()
