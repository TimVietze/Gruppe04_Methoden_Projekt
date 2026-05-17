from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data


@pytest.fixture(scope="module")
def price_snapshot():
    return data.load_price_snapshot()


@pytest.fixture(scope="module")
def risk_snapshot():
    return data.load_risk_snapshot()


def test_price_snapshot_basic_shape(price_snapshot):
    assert len(price_snapshot) > 0
    assert price_snapshot["CBSA_CODE"].notna().all()
    assert price_snapshot["price_now"].notna().all()
    assert price_snapshot["predicted_price_next_6m"].notna().all()
    assert price_snapshot["CBSA_CODE_STR"].str.len().eq(5).all()


def test_price_change_pct_matches_definition(price_snapshot):
    expected = (price_snapshot["predicted_price_next_6m"] - price_snapshot["price_now"]) / price_snapshot["price_now"]
    pd.testing.assert_series_equal(
        price_snapshot["price_change_pct"], expected, check_names=False
    )


def test_price_snapshot_has_all_9_categories(price_snapshot):
    expected = {
        "1br_mid", "2br_mid", "3br_mid", "4br_mid", "5br_mid",
        "sfr_mid", "condo_mid", "all_top", "all_bottom",
    }
    assert set(price_snapshot["category"].unique()) == expected


def test_risk_snapshot_one_row_per_cbsa(risk_snapshot):
    assert risk_snapshot["CBSA_CODE"].is_unique
    assert len(risk_snapshot) == 855


def test_risk_snapshot_band_values(risk_snapshot):
    allowed = {"Low", "Moderate", "High", "Very High"}
    assert set(risk_snapshot["damage_risk_band"].unique()).issubset(allowed)


def test_risk_percentile_in_range(risk_snapshot):
    assert risk_snapshot["damage_risk_percentile"].between(0, 100).all()


def test_build_combined_sfr_mid(price_snapshot, risk_snapshot):
    df = data.build_combined("sfr_mid", price_snapshot, risk_snapshot)
    assert len(df) > 0
    assert df["combined_score"].between(-100, 100).all()
    assert df["price_change_percentile"].between(0, 100).all()
    assert df["CBSA_CODE"].is_unique


def test_build_combined_score_symmetry(price_snapshot, risk_snapshot):
    df = data.build_combined("sfr_mid", price_snapshot, risk_snapshot)
    best = df.loc[df["combined_score"].idxmax()]
    worst = df.loc[df["combined_score"].idxmin()]
    assert best["combined_score"] > 0
    assert worst["combined_score"] < 0


def test_build_combined_unknown_category_raises(price_snapshot, risk_snapshot):
    with pytest.raises(ValueError):
        data.build_combined("nonexistent", price_snapshot, risk_snapshot)


def test_load_geojson_has_cbsa_code(price_snapshot):
    g = data.load_geojson()
    assert g["type"] == "FeatureCollection"
    assert len(g["features"]) > 0
    first_props = g["features"][0]["properties"]
    assert "CBSA_CODE" in first_props
    assert isinstance(first_props["CBSA_CODE"], str)


def test_category_table_returns_metro_categories(price_snapshot):
    sample_cbsa = int(price_snapshot["CBSA_CODE"].iloc[0])
    out = data.category_table(sample_cbsa, price_snapshot)
    assert len(out) > 0
    assert set(out.columns) == {"category", "price_now", "predicted_price_next_6m", "price_change_pct"}
