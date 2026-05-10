"""
Housing Market Attractiveness Predictor — Streamlit MVP

Reads pre-computed CSV outputs from app/data/:
  investment_ranking.csv  — 2026 growth forecasts for 855 US metros
  metrics.csv             — model comparison table
  predictions_test.csv    — actual vs predicted on 2022-2024 test set

Three tabs: Rankings | Model Performance | Limitations
"""

from pathlib import Path
import math

import pandas as pd
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).parent / "data"
RANKING_CSV   = DATA_DIR / "investment_ranking.csv"
METRICS_CSV   = DATA_DIR / "metrics.csv"
PRED_TEST_CSV = DATA_DIR / "predictions_test.csv"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Housing Market Attractiveness Predictor",
    page_icon="🏡",
    layout="wide",
)

# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_ranking() -> pd.DataFrame:
    df = pd.read_csv(RANKING_CSV, dtype={"CBSA_CODE": str})
    df = df.sort_values("predicted_growth_2026", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_CSV)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_TEST_CSV, dtype={"CBSA_CODE": str})
    # Map 0/1 to string so Streamlit uses categorical (not continuous) colors
    df["direction_label"] = df["direction_correct"].map({1: "Correct", 0: "Wrong"})
    return df


ranking_df = load_ranking()
metrics_df = load_metrics()
pred_df    = load_predictions()

# ── derive best model from metrics (highest dir-acc, tie-break lowest MAE) ────
candidates = (
    metrics_df[metrics_df["model"] != "Naive baseline"]
    .sort_values(["directional_accuracy", "mae_pct"], ascending=[False, True])
    .reset_index(drop=True)
)
best = candidates.iloc[0]

# ── header ────────────────────────────────────────────────────────────────────
st.title("Housing Market Attractiveness Predictor")
st.caption(
    "Predicting one-year-ahead mid-tier single-family home value growth "
    "across 855 US metropolitan areas"
)
st.info("Model-based signal only — not financial advice.")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_rankings, tab_perf, tab_limits = st.tabs(
    ["Rankings", "Model Performance", "Limitations"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Rankings
# ─────────────────────────────────────────────────────────────────────────────
with tab_rankings:
    st.subheader("2026 Growth Forecast — Metro Rankings")
    st.caption(
        "Mid-tier SFR growth is used as a proxy for regional owner-occupied housing market "
        "attractiveness, supporting investment decision-making by ranking US metropolitan areas."
    )

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        label_order = ["Attractive", "Moderate", "Neutral", "Caution"]
        selected_labels = st.multiselect(
            "Filter by recommendation",
            options=label_order,
            default=label_order,
        )
    with col_f2:
        search_text = st.text_input("Search metro by name", placeholder="e.g. Austin")

    # apply filters
    filtered = ranking_df.copy()
    if selected_labels:
        filtered = filtered[filtered["recommendation_label"].isin(selected_labels)]
    if search_text.strip():
        filtered = filtered[
            filtered["CBSA_TITLE"].str.contains(search_text.strip(), case=False, na=False)
        ]

    # summary metrics row
    counts = {lbl: int((filtered["recommendation_label"] == lbl).sum()) for lbl in label_order}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Metros shown", len(filtered))
    c2.metric("Attractive", counts["Attractive"])
    c3.metric("Moderate", counts["Moderate"])
    c4.metric("Neutral", counts["Neutral"])
    c5.metric("Caution", counts["Caution"])

    # top-20 bar chart
    top_n = min(20, len(filtered))
    if top_n > 0:
        top20 = filtered.head(top_n).copy()
        top20["metro_short"] = top20["CBSA_TITLE"].str.split(",").str[0]
        chart_data = top20.set_index("metro_short")[["predicted_growth_2026"]]
        st.markdown(f"#### Top {top_n} by Predicted Growth (2026)")
        st.bar_chart(chart_data, height=340)
        st.caption(
            "Predicted one-year-ahead mid-tier SFR growth (%). "
            "Ranked by model output."
        )
    else:
        st.info("No metros match the current filters.")

    # full filtered table
    st.markdown("#### Full Filtered Table")
    display_df = filtered[
        ["rank", "CBSA_TITLE", "predicted_growth_2026", "recommendation_label", "zhvi_sfr_mid_dec"]
    ].copy().rename(columns={
        "rank":                  "Rank",
        "CBSA_TITLE":            "Metro",
        "predicted_growth_2026": "Predicted Growth 2026 (%)",
        "recommendation_label":  "Recommendation",
        "zhvi_sfr_mid_dec":      "Current ZHVI ($)",
    })
    display_df["Predicted Growth 2026 (%)"] = display_df["Predicted Growth 2026 (%)"].map(
        lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
    )
    display_df["Current ZHVI ($)"] = display_df["Current ZHVI ($)"].map(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # climate & disaster expander — only shown when columns are present
    climate_cols = [
        c for c in ("had_major_disaster_ann", "damage_property_sum_ann")
        if c in filtered.columns
    ]
    if climate_cols:
        with st.expander("Climate & Disaster Risk Indicators"):
            st.markdown(
                "Weather and disaster variables (FEMA major disaster declarations, "
                "NOAA storm property damage) are included as **association-based risk "
                "indicators**, not causal drivers. At CBSA-annual aggregation, these "
                "features did **not** improve predictive performance over housing-history "
                "features alone — see the Model Performance tab for details. "
                "Do not interpret this table as evidence of a causal relationship between "
                "storm events and housing price changes."
            )
            risk_df = filtered[["rank", "CBSA_TITLE"] + climate_cols].copy().rename(columns={
                "rank":                    "Rank",
                "CBSA_TITLE":              "Metro",
                "had_major_disaster_ann":  "FEMA Major Disaster (2025)",
                "damage_property_sum_ann": "Storm Property Damage ($)",
            })
            if "Storm Property Damage ($)" in risk_df.columns:
                risk_df["Storm Property Damage ($)"] = risk_df["Storm Property Damage ($)"].map(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
                )
            st.dataframe(risk_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Model Performance — Test Set (2022–2024)")

    # best model callout
    st.success(
        f"**Selected best model: {best['model']} (Feature set {best['feature_set']})**  \n"
        f"Directional accuracy: {best['directional_accuracy']:.4f} · "
        f"MAE: {best['mae_pct']:.4f} pp  \n"
        "Selection criterion: highest directional accuracy, tie-break lowest MAE "
        "(Naive baseline excluded from selection)."
    )

    # metrics comparison table
    st.markdown("#### Metrics Comparison")
    metrics_display = metrics_df.copy().rename(columns={
        "model":                "Model",
        "feature_set":          "Feature Set",
        "mae_pct":              "MAE (pp)",
        "rmse_pct":             "RMSE (pp)",
        "r2":                   "R²",
        "directional_accuracy": "Directional Accuracy",
    })
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    st.caption(
        "Lower MAE / RMSE is better. Higher R² and directional accuracy are better. "
        "Negative R² on the test set is expected: mean training growth (~+5.5%) is "
        "substantially higher than post-COVID test growth (~+2.5%), causing a distribution "
        "shift that inflates the naive baseline. Directional accuracy (~80%) is the more "
        "meaningful metric in this context."
    )

    # weather/climate commentary derived directly from the metrics data
    a_best_acc = metrics_df.loc[metrics_df["feature_set"] == "A",      "directional_accuracy"].max()
    c_best_acc = metrics_df.loc[metrics_df["feature_set"] == "C-lite", "directional_accuracy"].max()
    a_best_mae = metrics_df.loc[metrics_df["feature_set"] == "A",      "mae_pct"].min()
    c_best_mae = metrics_df.loc[metrics_df["feature_set"] == "C-lite", "mae_pct"].min()

    if c_best_acc > a_best_acc or c_best_mae < a_best_mae:
        climate_note = (
            "Adding weather and disaster indicators (Feature set C-lite) produced a small change "
            "in performance metrics. These features are treated as association-based risk signals "
            "only — no causal relationship is claimed."
        )
    else:
        climate_note = (
            "Adding weather and disaster indicators (Feature set C-lite) did **not** improve "
            "performance over housing-history features alone (Feature set A). "
            "This is a valid scientific finding: at CBSA-annual aggregation, weather and disaster "
            "signals carry no additional predictive value beyond housing history."
        )
    st.markdown(f"> {climate_note}")

    # actual vs predicted scatter
    st.markdown("#### Actual vs. Predicted Growth — Test Set")

    year_options = sorted(int(y) for y in pred_df["YEAR"].unique())
    selected_year = st.selectbox(
        "Filter by test year", options=["All years"] + year_options
    )

    scatter_df = pred_df.copy()
    if selected_year != "All years":
        scatter_df = scatter_df[scatter_df["YEAR"] == selected_year]

    st.caption(
        f"Showing {len(scatter_df):,} observations. "
        "Green = direction predicted correctly, red = direction wrong."
    )
    st.scatter_chart(
        scatter_df,
        x="actual_growth",
        y="predicted_growth",
        color="direction_label",
        height=420,
    )

    # absolute error distribution — fixed bins, no sparse/noisy index
    st.markdown("#### Absolute Error Distribution")
    error_bins   = [0, 2, 4, 6, 8, float("inf")]
    error_labels = ["0–2", "2–4", "4–6", "6–8", "8+"]
    error_df = scatter_df.copy()
    error_df["error_bin"] = pd.cut(
        error_df["abs_error"], bins=error_bins, labels=error_labels, right=False
    )
    bin_counts = (
        error_df["error_bin"]
        .value_counts()
        .reindex(error_labels, fill_value=0)
        .rename("count")
    )
    st.bar_chart(bin_counts, height=260)
    st.caption("Distribution of |predicted − actual| in percentage points. Fixed-width bins.")

    # ── Ranking Backtest ──────────────────────────────────────────────────────
    st.markdown("#### Ranking Backtest (2022–2024)")
    st.markdown(
        "If the model had been used to rank metros by predicted growth, would higher-ranked "
        "metros have actually grown faster? The backtest below compares top-ranked, average, "
        "and bottom-ranked metros over the 2022–2024 test period."
    )

    _TOP_PCT = 0.10
    _bt_rows = []
    for _yr, _g in pred_df.groupby("YEAR"):
        _n  = len(_g)
        _k  = max(1, math.ceil(_n * _TOP_PCT))
        _gs = _g.sort_values("predicted_growth", ascending=False)
        _t  = _gs.head(_k)
        _b  = _gs.tail(_k)
        _bt_rows.append({
            "Year":                int(_yr),
            "Top 10% Mean (%)":    round(_t["actual_growth"].mean(), 2),
            "All Metros Mean (%)": round(_g["actual_growth"].mean(), 2),
            "Bot 10% Mean (%)":    round(_b["actual_growth"].mean(), 2),
            "Spread (pp)":         round(_t["actual_growth"].mean() - _b["actual_growth"].mean(), 2),
            "Hit Rate (%)":        round((_t["actual_growth"] > 0).mean() * 100, 1),
        })
    st.dataframe(pd.DataFrame(_bt_rows), use_container_width=True, hide_index=True)

    # overall summary
    _k_all   = max(1, math.ceil(len(pred_df) * _TOP_PCT))
    _df_rank = pred_df.sort_values("predicted_growth", ascending=False)
    _top_all = _df_rank.head(_k_all)
    _bot_all = _df_rank.tail(_k_all)
    _ov_top  = round(_top_all["actual_growth"].mean(), 2)
    _ov_all  = round(pred_df["actual_growth"].mean(), 2)
    _ov_bot  = round(_bot_all["actual_growth"].mean(), 2)
    _ov_spr  = round(_ov_top - _ov_bot, 2)
    _ov_hit  = round((_top_all["actual_growth"] > 0).mean() * 100, 1)
    st.info(
        f"**Overall (2022–2024 pooled, n={len(pred_df):,}):**  "
        f"Top 10% actual mean **{_ov_top:+.2f}%** · "
        f"All metros mean {_ov_all:+.2f}% · "
        f"Bottom 10% actual mean {_ov_bot:+.2f}%  \n"
        f"Top vs bottom spread **{_ov_spr:+.2f} pp** · "
        f"Top 10% hit rate (positive growth) **{_ov_hit:.1f}%**"
    )

    # decile chart
    st.markdown("**Actual mean growth by predicted decile (pooled, 2022–2024)**")
    _dec_temp = pred_df.copy()
    _dec_temp["pred_decile"] = pd.qcut(
        _dec_temp["predicted_growth"], 10, labels=False, duplicates="drop"
    )
    _decile_df = (
        _dec_temp.groupby("pred_decile", observed=True)["actual_growth"]
        .mean()
        .rename("Actual Mean Growth (%)")
        .round(3)
        .reset_index()
    )
    _decile_df["pred_decile"] = (_decile_df["pred_decile"] + 1).astype(int)
    _decile_df = _decile_df.rename(columns={"pred_decile": "Predicted Decile"})
    st.bar_chart(_decile_df.set_index("Predicted Decile"), height=300)
    st.caption(
        "Decile 1 = lowest predicted growth, Decile 10 = highest. "
        "A rising trend from left to right indicates ranking value."
    )

    st.markdown(
        "> The model shows screening value: historically, top-ranked metros outperformed "
        "bottom-ranked metros, and most top-ranked metros had positive growth. However, the "
        "signal is directional rather than precise; 2024 performance weakened due to post-COVID "
        "valuation corrections and missing economic variables such as mortgage rates, income, "
        "unemployment, and inflation."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Limitations
# ─────────────────────────────────────────────────────────────────────────────
with tab_limits:
    st.subheader("Limitations & Disclosures")

    st.markdown("""
**This tool is a research prototype. It is not financial advice.**

---

### Scope

- **Target:** Mid-tier single-family residential (SFR) home value growth, based on Zillow ZHVI SFR mid-tier data (2009–present).
- Condo markets, rental markets, luxury and distressed segments, and commercial real estate are **out of scope** and may behave differently.
- Predictions are at the **CBSA (metropolitan statistical area) level**. Neighbourhood-, ZIP-code-, and sub-metro variation is not captured.

---

### No Economic Variables

- The current model uses housing history only: current and prior-year ZHVI, and prior-year growth rate.
- Unemployment, income, mortgage rates, and inflation are **not included**. Their absence is the largest known gap in the model.
- Planned future extensions (priority order):
  1. BLS LAUS annual unemployment rate by CBSA
  2. Census ACS median household income by CBSA
  3. FRED mortgage rate (MORTGAGE30US) and CPI (CPIAUCSL)
  4. Confidence intervals via bootstrapped prediction or quantile regression

---

### Climate & Disaster Indicators

- Weather and disaster variables (FEMA major disaster declarations, NOAA storm property damage) are included as **association-based risk indicators only**.
- These features **did not improve model performance** at CBSA-annual aggregation — this is a valid scientific finding, not a gap.
- Do not interpret model outputs as evidence that storms or disasters cause housing price changes.

---

### Point Predictions Only

- The model produces a **single predicted growth value** per metro. There are no confidence intervals or prediction ranges.
- Rank order is indicative only — it is not a guarantee of future returns.
- The ranking backtest shows screening value, but the model should be interpreted as a directional ranking tool rather than a precise price-growth forecast.

---

### COVID-Era Distribution Shift

- Training years 2009–2021 include the pandemic price spike (+18–22% nationally in 2021).
- Mean training target: ~+5.5%. Mean test-period target: ~+2.5%.
- This shift causes **negative R² on the test set** — the model explains less variance than a mean predictor on out-of-sample data. This is documented and expected.
- Directional accuracy (~80%) is the more meaningful performance metric in this context.

---

### Recommendation Labels

These labels are fixed thresholds applied to the model's predicted growth output. They are **not buy or sell signals**.

| Label | Predicted Growth Threshold |
|---|---|
| Attractive | ≥ 6% |
| Moderate | 2% – 6% |
| Neutral | −2% – 2% |
| Caution | < −2% |

---

> **Model-based signal only — not financial advice.**
""")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Zillow ZHVI SFR mid-tier (2009–2025) · NOAA Storm Events · FEMA Disaster Declarations  |  "
    f"Best model: {best['model']} (Feature set {best['feature_set']})  |  "
    "Test period: 2022–2024"
)
