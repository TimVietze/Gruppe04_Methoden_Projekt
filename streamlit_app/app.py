"""ClimateHome — Investor View. Streamlit entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running with `streamlit run streamlit_app/app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import data
import viz
from text import CATEGORY_EN, METRIC_LABELS

st.set_page_config(
    page_title="ClimateHome — Investor View",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data(show_spinner="Loading price forecasts ...")
def _price():
    return data.load_price_snapshot()


@st.cache_data(show_spinner="Loading damage risk ...")
def _risk():
    return data.load_risk_snapshot()


@st.cache_data(show_spinner=False)
def _combined(category: str):
    return data.build_combined(category, _price(), _risk())


@st.cache_resource(show_spinner="Loading map data ...")
def _geo():
    return data.load_geojson()


st.title("🏠 ClimateHome — Investor View for U.S. Metros")
st.caption(
    "Forecast horizon: **March – August 2026** (6 months ahead of the February 2026 baseline). "
    "Prices: Ridge model. Damage risk: business risk score from the 6-month damage model."
)

with st.sidebar:
    st.header("Filters")
    available_categories = sorted(_price()["category"].unique())
    default_idx = available_categories.index("sfr_mid") if "sfr_mid" in available_categories else 0
    category = st.selectbox(
        "Housing category",
        options=available_categories,
        index=default_idx,
        format_func=lambda c: CATEGORY_EN.get(c, c),
    )
    metric = st.radio(
        "Map metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda m: METRIC_LABELS[m],
        index=0,
    )
    st.divider()
    st.caption(
        "**Investor Score** = price-change percentile − risk percentile. "
        "Range ≈ −100 (avoid) to +100 (very attractive)."
    )

combined = _combined(category)
combined = combined.sort_values("combined_score", ascending=False).reset_index(drop=True)
category_label = CATEGORY_EN.get(category, category)

st.subheader(f"Category: {category_label}")
viz.kpi_cards(combined, category_label)
st.divider()

map_col, detail_col = st.columns([7, 5])

with map_col:
    st.subheader(f"Map — {METRIC_LABELS[metric]}")
    st.plotly_chart(viz.map_choropleth(combined, metric, _geo()), use_container_width=True)

with detail_col:
    st.subheader("Metro detail")
    st.caption(f"Category: {category_label}")
    metro_options = combined["CBSA_CODE"].tolist()
    metro_labels = dict(zip(combined["CBSA_CODE"], combined["CBSA_TITLE"]))
    selected_cbsa = st.selectbox(
        "Select metro",
        options=metro_options,
        format_func=lambda c: f"{metro_labels[c]}  (Score {combined.loc[combined['CBSA_CODE']==c, 'combined_score'].iloc[0]:+.1f})",
        index=0,
    )
    row = combined.loc[combined["CBSA_CODE"] == selected_cbsa].iloc[0]
    viz.detail_headline(row)
    viz.detail_verdict(row)
    st.markdown("**Risk breakdown**")
    viz.detail_risk_bars(row)
    st.markdown("**All categories for this metro**")
    viz.detail_category_table(data.category_table(int(selected_cbsa), _price()))

st.divider()
viz.ranking_tables(combined)

with st.expander("Data quality & notes"):
    n_geo_missing = combined[~combined["CBSA_CODE_STR"].isin(
        {f["properties"]["CBSA_CODE"] for f in _geo()["features"]}
    )]
    st.write(
        f"- Category **{category_label}**: {len(combined)} of 855 metros have price data."
    )
    st.write(
        f"- {len(n_geo_missing)} of those metros are missing from the map geometry (Census 1:20m)."
    )
    st.write(
        "- Damage risk uses the Feb 2026 baseline row from the 6-month damage model "
        "(cumulative forecast over Mar – Aug 2026)."
    )
