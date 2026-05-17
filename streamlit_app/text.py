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

    Three score regimes — high (≥ 60), low (≤ −40), and tepid (|score| < 20) —
    each branch differentiates by the specific risk band so Low vs Moderate
    and High vs Very High get distinct wording.
    """
    if combined_score >= 60:
        if risk_band == "Low":
            return "Strong return potential at low risk — very attractive market."
        if risk_band == "Moderate":
            return "Strong return potential at moderate risk — attractive market."
        if risk_band == "High":
            return "High return potential, but elevated damage risk — review carefully."
        if risk_band == "Very High":
            return "High return potential, but very high damage risk — review very carefully."

    if combined_score <= -40:
        if risk_band == "Very High":
            return "Limited upside paired with very high damage risk — strongly avoid."
        if risk_band == "High":
            return "Limited upside paired with elevated risk — avoid."
        if risk_band == "Moderate":
            return "Limited upside at moderate risk — better to avoid."
        if risk_band == "Low":
            return "Limited upside, though risk is low — better to avoid."

    if abs(combined_score) < 20:
        if risk_band == "Very High":
            return "Low upside paired with very high risk — clearly unattractive."
        if risk_band == "High":
            return "Low upside paired with elevated risk — unattractive."

    return "Average market — no clear recommendation."
