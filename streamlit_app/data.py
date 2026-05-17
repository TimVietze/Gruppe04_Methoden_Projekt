"""Data loading and snapshot construction. Pure pandas — no Streamlit imports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Pandas 3.x defaults to pyarrow-backed string dtypes, which Streamlit's frontend
# serializes as "LargeUtf8" — a type its JS renderer does not understand and which
# crashes st.dataframe. Force object-dtype strings instead.
try:
    pd.options.future.infer_string = False
except (AttributeError, KeyError):
    pass

ROOT = Path(__file__).resolve().parents[1]
PRICE_CSV = (
    ROOT
    / "Model Analysis"
    / "Prediction absolute price 6 months"
    / "3 Results"
    / "prices as feature"
    / "results prices"
    / "absolute_predictions_ridge.csv"
)
RISK_CSV = (
    ROOT
    / "Model Analysis"
    / "Prediction Damage Property 6 month"
    / "results_damage_predictions_6m_with_business_risk_score.csv"
)
GEOJSON = ROOT / "Methoden Data" / "Geo Data" / "CBSA_20m.geojson"

PRICE_AS_OF = "2026-02-01"  # 6-month-ahead target = Aug 2026 (latest baseline)
RISK_AS_OF = "2026-02"      # damage forecast for the same Mar–Aug 2026 window


def _cbsa_str(s: pd.Series) -> pd.Series:
    return s.astype(int).astype(str).str.zfill(5)


def load_price_snapshot() -> pd.DataFrame:
    """One row per (CBSA, category) for the latest as-of date in the ridge file."""
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Price CSV not found: {PRICE_CSV}")
    df = pd.read_csv(PRICE_CSV)
    df = df[df["YEAR_MONTH"] == PRICE_AS_OF].copy()
    if df.empty:
        raise ValueError(f"No rows in price CSV for YEAR_MONTH={PRICE_AS_OF}")
    df["CBSA_CODE_STR"] = _cbsa_str(df["CBSA_CODE"])
    df["price_change_pct"] = (
        (df["predicted_price_next_6m"] - df["price_now"]) / df["price_now"]
    )
    return df[
        [
            "CBSA_CODE",
            "CBSA_CODE_STR",
            "CBSA_TITLE",
            "category",
            "price_now",
            "predicted_price_next_6m",
            "price_change_pct",
        ]
    ].reset_index(drop=True)


def load_risk_snapshot() -> pd.DataFrame:
    """One row per CBSA at YEAR_MONTH = RISK_AS_OF.

    The damage model's `expected_damage_6m` at month T is its cumulative
    forecast over [T, T+6m). Picking the latest available baseline gives the
    most forward-looking risk view that aligns with the price horizon.
    """
    if not RISK_CSV.exists():
        raise FileNotFoundError(f"Risk CSV not found: {RISK_CSV}")
    df = pd.read_csv(RISK_CSV)
    snap = df[df["YEAR_MONTH"] == RISK_AS_OF].copy()
    if snap.empty:
        raise ValueError(f"Risk CSV has no rows for YEAR_MONTH={RISK_AS_OF}")
    snap = snap.rename(
        columns={
            "damage_risk_percentile_6m": "damage_risk_percentile",
            "damage_risk_band_6m": "damage_risk_band",
        }
    )
    out = snap[
        [
            "CBSA_CODE",
            "CBSA_TITLE",
            "damage_risk_percentile",
            "damage_risk_band",
            "predicted_damage_probability",
            "predicted_damage_severity_dollars",
            "expected_damage_6m",
        ]
    ].copy()
    out["CBSA_CODE_STR"] = _cbsa_str(out["CBSA_CODE"])
    return out.reset_index(drop=True)


def build_combined(
    category: str,
    price_snapshot: pd.DataFrame,
    risk_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """One row per metro for the chosen category.

    Adds price_change_percentile (ranked within the category, since absolute
    % changes vary by category) and combined_score = price_pctl − risk_pctl.
    """
    p = price_snapshot[price_snapshot["category"] == category].copy()
    if p.empty:
        raise ValueError(f"No price rows for category={category}")
    p["price_change_percentile"] = p["price_change_pct"].rank(pct=True) * 100

    r = risk_snapshot[
        [
            "CBSA_CODE",
            "damage_risk_percentile",
            "damage_risk_band",
            "predicted_damage_probability",
            "predicted_damage_severity_dollars",
            "expected_damage_6m",
        ]
    ]
    merged = p.merge(r, on="CBSA_CODE", how="inner")
    merged["combined_score"] = (
        merged["price_change_percentile"] - merged["damage_risk_percentile"]
    )
    return merged.reset_index(drop=True)


def load_geojson() -> dict:
    if not GEOJSON.exists():
        raise FileNotFoundError(
            f"Geojson not found: {GEOJSON}. Run Data Cleaning Scripts/6_CBSA_Geodata.py to build it."
        )
    with open(GEOJSON) as f:
        return json.load(f)


def category_table(cbsa_code: int, price_snapshot: pd.DataFrame) -> pd.DataFrame:
    """All 9 categories for one metro, formatted for display."""
    df = price_snapshot[price_snapshot["CBSA_CODE"] == cbsa_code].copy()
    return df[["category", "price_now", "predicted_price_next_6m", "price_change_pct"]].sort_values(
        "category"
    ).reset_index(drop=True)
