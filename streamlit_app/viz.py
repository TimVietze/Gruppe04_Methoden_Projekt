"""Plotly figures + Streamlit-aware display helpers for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from text import BAND_EN, CATEGORY_EN, METRIC_LABELS, verdict

METRIC_COLORSCALES = {
    "combined_score": "RdYlGn",
    "price_change_pct": "RdYlGn",
    "damage_risk_percentile": "RdYlGn_r",
}


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Force string-like columns to numpy object dtype so Streamlit's frontend
    can serialize them. Pandas 3.x defaults to pyarrow string arrays that
    serialize as Arrow "LargeUtf8", which the Streamlit JS renderer rejects."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col]
        dt = str(s.dtype).lower()
        if s.dtype.kind in {"O", "U"} or "string" in dt or "utf8" in dt or dt == "str":
            out[col] = pd.array([None if v is None else str(v) for v in s.tolist()], dtype="object")
        else:
            out[col] = s.values
    return out


def map_choropleth(df: pd.DataFrame, metric: str, geojson: dict) -> go.Figure:
    """Choropleth of metros colored by the chosen metric (uses go.Choroplethmapbox)."""
    hover_band = df["damage_risk_band"].map(BAND_EN).fillna(df["damage_risk_band"])
    custom = list(
        zip(
            df["CBSA_TITLE"],
            df["price_change_pct"] * 100,
            df["damage_risk_percentile"],
            hover_band,
            df["combined_score"],
        )
    )
    zmin, zmax = _range_for(df, metric)
    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=df["CBSA_CODE_STR"],
            z=df[metric],
            featureidkey="properties.CBSA_CODE",
            colorscale=METRIC_COLORSCALES[metric],
            zmin=zmin,
            zmax=zmax,
            marker_opacity=0.7,
            marker_line_width=0.3,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pred. price change: %{customdata[1]:.2f}%<br>"
                "Risk percentile: %{customdata[2]:.1f}<br>"
                "Risk band: %{customdata[3]}<br>"
                "Investor score: %{customdata[4]:.1f}<extra></extra>"
            ),
            colorbar={"title": METRIC_LABELS[metric]},
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 39.0, "lon": -97.5},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=550,
    )
    return fig


def _range_for(df: pd.DataFrame, metric: str) -> list[float]:
    if metric == "combined_score":
        return [-100, 100]
    if metric == "damage_risk_percentile":
        return [0, 100]
    s = df["price_change_pct"]
    bound = max(abs(s.min()), abs(s.max()))
    return [-bound, bound]


def kpi_cards(df: pd.DataFrame, category_label: str) -> None:
    cols = st.columns(4)
    n = len(df)
    avg_pct = df["price_change_pct"].mean() * 100
    very_high_share = (df["damage_risk_band"] == "Very High").mean() * 100
    top_row = df.loc[df["combined_score"].idxmax()]

    cols[0].metric("Metros with data", f"{n}")
    cols[1].metric("Avg. predicted price change (6M)", f"{avg_pct:+.2f}%")
    cols[2].metric("Share with 'Very High' risk", f"{very_high_share:.1f}%")
    cols[3].metric(
        f"Top market ({category_label})",
        top_row["CBSA_TITLE"],
        f"Score {top_row['combined_score']:+.1f}",
    )


def _format_ranking(slice_: pd.DataFrame) -> pd.DataFrame:
    out = slice_[["CBSA_TITLE", "combined_score", "price_change_pct", "damage_risk_band"]].copy()
    out["combined_score"] = out["combined_score"].map(lambda v: f"{v:+.1f}")
    out["price_change_pct"] = out["price_change_pct"].map(lambda v: f"{v:+.2%}")
    out["damage_risk_band"] = out["damage_risk_band"].map(BAND_EN).fillna(out["damage_risk_band"])
    return out.rename(
        columns={
            "CBSA_TITLE": "Metro",
            "combined_score": "Investor Score",
            "price_change_pct": "Price Change (6M)",
            "damage_risk_band": "Risk",
        }
    ).reset_index(drop=True)


def ranking_tables(df: pd.DataFrame) -> None:
    top = _format_ranking(df.nlargest(10, "combined_score"))
    bottom = _format_ranking(df.nsmallest(10, "combined_score"))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 — Best Markets")
        st.dataframe(_arrow_safe(top), hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Top 10 — Avoid")
        st.dataframe(_arrow_safe(bottom), hide_index=True, use_container_width=True)


def detail_headline(row: pd.Series) -> None:
    price_cols = st.columns(3)
    price_cols[0].metric("Current price", f"${row['price_now']:,.0f}")
    price_cols[1].metric(
        "Forecast (6M)", f"${row['predicted_price_next_6m']:,.0f}",
        f"{row['price_change_pct']:+.2%}",
    )
    price_cols[2].metric("Investor Score", f"{row['combined_score']:+.1f}")

    risk_cols = st.columns(2)
    risk_cols[0].metric(
        "Risk band",
        BAND_EN.get(row["damage_risk_band"], row["damage_risk_band"]),
    )
    risk_cols[1].metric("Expected damage (6M)", f"${row['expected_damage_6m']:,.0f}")


def detail_verdict(row: pd.Series) -> None:
    text = verdict(row["combined_score"], row["damage_risk_band"], row["price_change_pct"])
    st.info(f"**Investor verdict:** {text}")


def detail_risk_bars(row: pd.Series) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[row["predicted_damage_probability"] * 100],
            y=["Damage probability (%)"],
            orientation="h",
            marker_color="#d62728",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[row["damage_risk_percentile"]],
            y=["Risk percentile"],
            orientation="h",
            marker_color="#ff7f0e",
        )
    )
    fig.update_layout(
        height=180,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
        xaxis={"range": [0, 100], "title": "0 – 100"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Expected damage severity (given an event): ${row['predicted_damage_severity_dollars']:,.0f}"
    )


def detail_category_table(df9: pd.DataFrame) -> None:
    if df9.empty:
        st.warning("No category data for this metro.")
        return
    out = df9.copy()
    out["category"] = out["category"].map(CATEGORY_EN).fillna(out["category"])
    out["price_now"] = out["price_now"].map(lambda v: f"${v:,.0f}")
    out["predicted_price_next_6m"] = out["predicted_price_next_6m"].map(lambda v: f"${v:,.0f}")
    out["price_change_pct"] = out["price_change_pct"].map(lambda v: f"{v:+.2%}")
    out = out.rename(
        columns={
            "category": "Category",
            "price_now": "Current price",
            "predicted_price_next_6m": "Forecast (6M)",
            "price_change_pct": "Change",
        }
    )
    st.dataframe(_arrow_safe(out), hide_index=True, use_container_width=True)
