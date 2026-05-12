"""
Housing Market Attractiveness Predictor — Streamlit MVP

Reads pre-computed CSV outputs from app/data/:
  investment_ranking.csv  — 2026 growth forecasts for 855 US metros
  metrics.csv             — model comparison table
  predictions_test.csv    — actual vs predicted on 2022-2024 test set

Three tabs: Opportunity Screening | Model Evidence | Limitations
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
    df["dollar_upside"]     = df["zhvi_sfr_mid_dec"] * df["predicted_growth_2026"] / 100
    df["abs_dollar_upside"] = df["dollar_upside"].abs()
    df["disaster_label"] = df["had_major_disaster_ann"].map(
        {0: "No Disaster Flag", 1: "Disaster Flagged"}
    ).fillna("No Disaster Flag")
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

# ── Spearman ρ via pandas rank correlation (no scipy required) ─────────────────
_rho = pred_df["predicted_growth"].rank().corr(pred_df["actual_growth"].rank())

# ── header ────────────────────────────────────────────────────────────────────
st.title("Housing Market Attractiveness Predictor")
st.caption(
    "Predicting one-year-ahead mid-tier single-family home value growth "
    "across 855 US metropolitan areas"
)

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_rankings, tab_perf, tab_limits = st.tabs(
    ["Opportunity Screening", "Model Evidence", "Limitations"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Opportunity Screening
# ─────────────────────────────────────────────────────────────────────────────
with tab_rankings:
    st.subheader("2026 Screening Rankings")
    st.caption(
        "Mid-tier SFR growth is used as a proxy for regional owner-occupied housing market "
        "attractiveness, supporting investment decision-making by ranking US metropolitan areas."
    )

    # ── filters ───────────────────────────────────────────────────────────────
    label_order = ["Attractive", "Moderate", "Neutral", "Caution"]
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
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

    # ── six KPI cards ──────────────────────────────────────────────────────────
    counts  = {lbl: int((filtered["recommendation_label"] == lbl).sum()) for lbl in label_order}
    attr_f  = filtered[filtered["recommendation_label"] == "Attractive"]
    med_growth  = f"{attr_f['predicted_growth_2026'].median():+.1f}%" if len(attr_f) > 0 else "—"
    med_upside  = f"${attr_f['dollar_upside'].median():,.0f}" if len(attr_f) > 0 else "—"
    disaster_ct = int((filtered["had_major_disaster_ann"] == 1).sum()) \
                  if "had_major_disaster_ann" in filtered.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Metros Shown",              len(filtered))
    c2.metric("Attractive",                counts["Attractive"])
    c3.metric("Caution",                   counts["Caution"])
    c4.metric("Median Growth (Attractive)", med_growth)
    c5.metric("Median Dollar Upside (Attractive)", med_upside)
    c6.metric("Metros with Disaster Flag",  disaster_ct)

    st.info("Model-based signal only — not financial advice.")
    st.warning(
        "**Systematic overestimate (2022–2024 test period):** "
        "Model predictions averaged **3.7 percentage points above realized outcomes**. "
        "Use rankings as directional screening signals — not precise point forecasts."
    )

    # ── top-20 bar chart ──────────────────────────────────────────────────────
    top_n = min(20, len(filtered))
    if top_n > 0:
        top20 = filtered.head(top_n).copy()
        top20["metro_short"] = top20["CBSA_TITLE"].str.split(",").str[0]
        chart_data = top20.set_index("metro_short")[["predicted_growth_2026"]]
        st.markdown(f"#### Top {top_n} Metros by Predicted Mid-Tier SFR Growth")
        st.bar_chart(chart_data, height=340)
        st.caption(
            "Predicted one-year-ahead mid-tier SFR growth (%). Ranked by model output. "
            "Implied dollar change varies by current home price — see the table below."
        )
    else:
        st.info("No metros match the current filters.")

    # ── Opportunity-Risk scatter ───────────────────────────────────────────────
    if len(filtered) > 0:
        st.markdown("#### Growth–Price Screening View")
        _excl_disaster = st.checkbox("Exclude disaster-flagged metros", value=False)
        _scatter_data = filtered.copy()
        if _excl_disaster and "had_major_disaster_ann" in _scatter_data.columns:
            _scatter_data = _scatter_data[_scatter_data["had_major_disaster_ann"] != 1]

        st.scatter_chart(
            _scatter_data,
            x="predicted_growth_2026",
            y="zhvi_sfr_mid_dec",
            color="disaster_label",
            size="abs_dollar_upside",
            x_label="Predicted Growth 2026 (%)",
            y_label="Current ZHVI Mid-Tier ($)",
            height=420,
        )
        st.caption(
            "Each point is a metro. Color = disaster flag status. "
            "Bubble size = absolute implied dollar change. "
            "Metros on the right have higher predicted growth; lower points have lower current entry price levels. "
            "This view helps compare growth signal, price level, dollar magnitude, and disaster-risk context "
            "in one place. It is a directional screening view, not a recommendation."
        )

        # ── split tables: Attractive clean vs disaster-flagged ────────────────
        st.markdown("#### Attractive Metros: Clean vs. Disaster-Flagged")
        _attr = filtered[filtered["recommendation_label"] == "Attractive"].copy()
        _split_cols = ["CBSA_TITLE", "predicted_growth_2026", "dollar_upside", "zhvi_sfr_mid_dec"]
        _split_rename = {
            "CBSA_TITLE":            "Metro",
            "predicted_growth_2026": "Predicted Growth (%)",
            "dollar_upside":         "Implied $ Change",
            "zhvi_sfr_mid_dec":      "Current ZHVI ($)",
        }

        if "had_major_disaster_ann" in _attr.columns:
            _clean   = _attr[_attr["had_major_disaster_ann"] != 1][_split_cols].copy()
            _flagged = _attr[_attr["had_major_disaster_ann"] == 1][_split_cols].copy()
        else:
            _clean, _flagged = _attr[_split_cols].copy(), _attr[_split_cols].iloc[:0].copy()

        _clean   = _clean.sort_values("predicted_growth_2026", ascending=False).head(10)
        _flagged = _flagged.sort_values("predicted_growth_2026", ascending=False).head(10)

        for _df in [_clean, _flagged]:
            _df.rename(columns=_split_rename, inplace=True)
            _df["Predicted Growth (%)"] = _df["Predicted Growth (%)"].map(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
            )
            _df["Implied $ Change"] = _df["Implied $ Change"].map(
                lambda x: (f"+${x:,.0f}" if x >= 0 else f"−${abs(x):,.0f}") if pd.notna(x) else "—"
            )
            _df["Current ZHVI ($)"] = _df["Current ZHVI ($)"].map(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
            )

        _n_clean   = len(_attr[_attr["had_major_disaster_ann"] != 1]) if "had_major_disaster_ann" in _attr.columns else len(_attr)
        _n_flagged = len(_attr[_attr["had_major_disaster_ann"] == 1]) if "had_major_disaster_ann" in _attr.columns else 0

        _col_l, _col_r = st.columns(2)
        with _col_l:
            st.markdown(f"**Attractive — No Disaster Flag** ({_n_clean} metros, top 10 shown)")
            st.dataframe(_clean, use_container_width=True, hide_index=True)
        with _col_r:
            st.markdown(f"**Attractive — Disaster Flagged** ({_n_flagged} metros, top 10 shown)")
            if len(_flagged) > 0:
                st.dataframe(_flagged, use_container_width=True, hide_index=True)
            else:
                st.info("No disaster-flagged metros in current filter.")

    # ── climate & disaster risk context (visible) ─────────────────────────────
    st.markdown("#### Climate & Disaster Risk Context")
    _exp_base = filtered.copy()
    if len(_exp_base) == 0:
        st.info("No metros match the current filters.")
    elif "had_major_disaster_ann" in _exp_base.columns:
        _dis_grp  = _exp_base[_exp_base["had_major_disaster_ann"] == 1]["predicted_growth_2026"]
        _ndis_grp = _exp_base[_exp_base["had_major_disaster_ann"] != 1]["predicted_growth_2026"]
        _dis_mean  = _dis_grp.mean()
        _ndis_mean = _ndis_grp.mean()
        _exp_m1, _exp_m2 = st.columns(2)
        _exp_m1.metric(
            "Avg Predicted Growth — No Disaster Flag",
            f"{_ndis_mean:.2f}%" if pd.notna(_ndis_mean) else "—",
        )
        _exp_m2.metric(
            "Avg Predicted Growth — Disaster Flagged",
            f"{_dis_mean:.2f}%" if pd.notna(_dis_mean) else "—",
            delta=f"{_dis_mean - _ndis_mean:+.2f} pp vs. no-disaster" if pd.notna(_dis_mean) and pd.notna(_ndis_mean) else None,
        )

        _ct = pd.crosstab(
            _exp_base["recommendation_label"],
            _exp_base["had_major_disaster_ann"].map({0: "No Disaster", 1: "Disaster"}),
        )
        _ct = _ct.reindex(["Attractive", "Moderate", "Neutral", "Caution"]).dropna(how="all")
        st.markdown(f"**Recommendation × Disaster Flag ({len(_exp_base)} metros in current filter)**")
        st.dataframe(_ct, use_container_width=True)

    with st.expander("About these indicators and the market opportunity hypothesis"):
        st.markdown(
            "**Disaster indicators are risk context, not a predictive signal.** "
            "FEMA major disaster declarations and NOAA storm property damage are included as "
            "**association-based risk indicators**, not causal drivers of price changes. "
            "In the current model, adding these variables (Feature set C-lite) did **not** "
            "improve predictive performance over housing history alone at CBSA-annual aggregation. "
            "Use the Disaster Flag column as background context when evaluating a metro, "
            "not as a ranking input."
        )
        st.markdown(
            "**Market opportunity hypothesis:** If disaster risk suppresses demand and depresses "
            "prices in affected metros, a high predicted growth in a disaster-flagged area could "
            "indicate an undervalued market recovering from a shock — or it could reflect model "
            "limitations. The model does not distinguish between these interpretations. "
            "Treat disaster-flagged metros as warranting additional due diligence, "
            "not automatic exclusion."
        )

    # ── full filtered table (collapsed) ───────────────────────────────────────
    with st.expander("Full Ranking Table — all filtered metros"):
        table_cols = [
            "rank", "CBSA_TITLE", "predicted_growth_2026", "dollar_upside",
            "zhvi_sfr_mid_dec", "recommendation_label",
        ]
        if "had_major_disaster_ann" in filtered.columns:
            table_cols.append("had_major_disaster_ann")

        display_df = filtered[table_cols].copy().rename(columns={
            "rank":                  "Rank",
            "CBSA_TITLE":            "Metro",
            "predicted_growth_2026": "Predicted Growth 2026 (%)",
            "dollar_upside":         "Implied Dollar Change ($)",
            "zhvi_sfr_mid_dec":      "Current ZHVI ($)",
            "recommendation_label":  "Recommendation",
            "had_major_disaster_ann": "Disaster Flag",
        })
        display_df["Predicted Growth 2026 (%)"] = display_df["Predicted Growth 2026 (%)"].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        display_df["Implied Dollar Change ($)"] = display_df["Implied Dollar Change ($)"].map(
            lambda x: (f"+${x:,.0f}" if x >= 0 else f"−${abs(x):,.0f}") if pd.notna(x) else "—"
        )
        display_df["Current ZHVI ($)"] = display_df["Current ZHVI ($)"].map(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
        if "Disaster Flag" in display_df.columns:
            display_df["Disaster Flag"] = display_df["Disaster Flag"].map(
                lambda x: "⚠ Yes" if x == 1 else "—"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Evidence
# ─────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("Model Performance — Test Set (2022–2024)")
    st.markdown(
        "The ranking backtest is the primary evidence that this model creates value as a "
        "directional screening tool. The technical metrics below document its accuracy boundaries."
    )

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
    st.caption(
        "Hit rate = % of top-10% predicted metros with positive actual growth. "
        "In 2024, ~64% of all metros had positive growth — the hit rate of 67.4% was "
        "only ~3 pp above the base rate that year, the weakest test year."
    )

    # overall summary callout
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

    # decile chart — business-value visualization
    st.markdown(
        "**Did higher predicted-growth groups actually achieve higher realized growth "
        "in the historical test period?**"
    )
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
        "An upward trend across deciles means that metros the model ranked higher actually "
        "achieved higher realized growth — supporting the ranking logic. "
        "The slight dip at decile 10 is consistent with the model's known tendency to "
        "overestimate the top-ranked metros due to COVID-era training bias."
    )

    st.markdown(
        "> The model shows screening value: historically, top-ranked metros outperformed "
        "bottom-ranked metros, and most top-ranked metros had positive growth. However, the "
        "signal is directional rather than precise; 2024 performance weakened due to post-COVID "
        "valuation corrections and missing economic variables such as mortgage rates, income, "
        "unemployment, and inflation."
    )

    # Spearman rank correlation
    st.markdown(
        f"**Rank correlation (Spearman ρ = {_rho:.3f}, n = {len(pred_df):,}):** "
        "The rank correlation between predicted and actual growth is positive, indicating that "
        "the model has directional screening value, even though exact point predictions remain noisy."
    )

    # ── Feature-set comparison table ──────────────────────────────────────────
    st.markdown("#### Feature Set Comparison — Directional Accuracy")
    _naive_acc = metrics_df.loc[
        metrics_df["model"] == "Naive baseline",
        "directional_accuracy"
    ].iloc[0]

    _fsc = metrics_df.copy()
    _fsc["Δ dir. acc. vs. naive (pp)"] = (
        (_fsc["directional_accuracy"] - _naive_acc) * 100
    ).round(2)

    _fsc_display = _fsc.rename(columns={
        "model":                "Model",
        "feature_set":          "Feature Set",
        "directional_accuracy": "Dir. Accuracy",
        "mae_pct":              "MAE (pp)",
    })[["Model", "Feature Set", "Dir. Accuracy", "MAE (pp)", "Δ dir. acc. vs. naive (pp)"]]

    st.dataframe(_fsc_display, use_container_width=True, hide_index=True)
    st.caption(
        "Δ = directional accuracy gain over the naive baseline, which predicts the same growth rate for every metro. "
        "Ridge [A] (housing history only) achieves the best directional accuracy. "
        "Adding weather/disaster features [C-lite] does not improve over housing history alone — "
        "a valid scientific finding at CBSA-annual aggregation."
    )

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("#### Model Metrics — Test Set (2022–2024)")
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
        "Negative R² is expected: mean training growth (~+5.5%) is substantially higher "
        "than post-COVID test growth (~+2.5%), causing a distribution shift. "
        "Directional accuracy (~80%) is the more meaningful metric for a ranking use case."
    )
    st.markdown(
        "> **Ridge vs. naive baseline:** The naive baseline has lower MAE, but it predicts "
        "the same growth rate for every metro. Ridge is selected because it has the highest "
        "directional accuracy, which better matches the ranking/screening use case."
    )

    # weather/climate commentary derived from metrics data
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

    # ── Actual vs. Predicted scatter ───────────────────────────────────────────
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
        "Blue = model correctly predicted whether a metro would grow or decline. "
        "Red = direction predicted incorrectly. "
        "Points along the diagonal would indicate perfect predictions."
    )
    st.scatter_chart(
        scatter_df,
        x="actual_growth",
        y="predicted_growth",
        x_label="Actual Growth (%)",
        y_label="Predicted Growth (%)",
        color="direction_label",
        height=420,
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
- In the 2022–2024 test period, predictions averaged roughly 3–4 percentage points above actual outcomes, likely due to COVID-era training data and missing macroeconomic variables.
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
