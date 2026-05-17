"""English labels and verdict generator. Framework-free, easily unit-testable."""

from __future__ import annotations

BAND_EN = {
    "Low": "Low",
    "Moderate": "Moderate",
    "High": "High",
    "Very High": "Very High",
}

BAND_ORDER = ["Low", "Moderate", "High", "Very High"]
HIGH_RISK_BANDS = {"High", "Very High"}

CATEGORY_EN = {
    "1br_mid": "1-Bedroom (mid-tier)",
    "2br_mid": "2-Bedroom (mid-tier)",
    "3br_mid": "3-Bedroom (mid-tier)",
    "4br_mid": "4-Bedroom (mid-tier)",
    "5br_mid": "5-Bedroom (mid-tier)",
    "sfr_mid": "Single-Family Home (mid-tier)",
    "condo_mid": "Condominium (mid-tier)",
    "all_top": "All Homes (top tier)",
    "all_bottom": "All Homes (bottom tier)",
}

METRIC_LABELS = {
    "combined_score": "Investor Score (Return − Risk)",
    "price_change_pct": "Predicted Price Change (6M)",
    "damage_risk_percentile": "Climate Damage Risk (percentile)",
}


def verdict(combined_score: float, risk_band: str, price_change_pct: float) -> str:
    """Return a short English investor verdict.

    Buckets (checked in order):
    1. score ≥ 60 and risk low/moderate → strong buy
    2. score ≥ 60 and risk high/very high → high-reward-high-risk
    3. score ≤ −40 → avoid
    4. abs(score) < 20 and risk high/very high → risky stagnation
    5. fallback → neutral
    """
    if combined_score >= 60 and risk_band not in HIGH_RISK_BANDS:
        return "Strong return potential at moderate risk — attractive market."
    if combined_score >= 60 and risk_band in HIGH_RISK_BANDS:
        return "High return potential, but very high damage risk — review carefully."
    if combined_score <= -40:
        return "Limited upside — better to avoid."
    if abs(combined_score) < 20 and risk_band in HIGH_RISK_BANDS:
        return "Low upside paired with elevated risk — unattractive."
    return "Average market — no clear recommendation."
