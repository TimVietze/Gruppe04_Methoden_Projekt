from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text import BAND_EN, CATEGORY_EN, HIGH_RISK_BANDS, METRIC_LABELS, verdict


def test_band_en_covers_all_observed_bands():
    observed = {"Low", "Moderate", "High", "Very High"}
    assert observed.issubset(BAND_EN.keys())
    for v in BAND_EN.values():
        assert isinstance(v, str) and v


def test_category_en_covers_all_9_categories():
    expected = {
        "1br_mid", "2br_mid", "3br_mid", "4br_mid", "5br_mid",
        "sfr_mid", "condo_mid", "all_top", "all_bottom",
    }
    assert expected == set(CATEGORY_EN.keys())


def test_metric_labels_keys():
    assert set(METRIC_LABELS.keys()) == {"combined_score", "price_change_pct", "damage_risk_percentile"}


def test_verdict_strong_buy_when_high_score_low_risk():
    msg = verdict(70.0, "Low", 0.05)
    assert "attractive" in msg.lower()


def test_verdict_high_reward_high_risk():
    msg = verdict(70.0, "Very High", 0.05)
    assert "carefully" in msg.lower()


def test_verdict_avoid_when_score_very_negative():
    msg = verdict(-60.0, "Low", -0.02)
    assert "avoid" in msg.lower()


def test_verdict_risky_stagnation():
    msg = verdict(5.0, "High", 0.0)
    assert "unattractive" in msg.lower()


def test_verdict_neutral_default():
    msg = verdict(0.0, "Low", 0.0)
    assert "average" in msg.lower()


def test_high_risk_bands_set():
    assert HIGH_RISK_BANDS == {"High", "Very High"}
