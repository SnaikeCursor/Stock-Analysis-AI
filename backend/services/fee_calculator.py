"""Swissquote-style fee estimate for Swiss stocks (SIX) — tier + platform surcharge.

Reference: tiered commission by trade volume (CHF) + 0.85% platform fee.
"""

from __future__ import annotations

PLATFORM_SURCHARGE_PCT = 0.0085  # 0.85%


def _tier_base_fee_chf(volume_chf: float) -> float:
    """Fixed tier component before platform surcharge."""
    v = float(volume_chf)
    if v <= 0:
        return 0.0
    if v < 500:
        return 3.0
    if v < 1000:
        return 5.0
    if v < 2000:
        return 10.0
    if v < 10000:
        return 30.0
    if v < 15000:
        return 55.0
    if v < 25000:
        return 80.0
    if v < 50000:
        return 135.0
    return 190.0


def swissquote_fee(volume_chf: float) -> float:
    """Total fee in CHF: tiered base + 0.85% of notional volume."""
    v = max(0.0, float(volume_chf))
    base = _tier_base_fee_chf(v)
    surcharge = v * PLATFORM_SURCHARGE_PCT
    return round(base + surcharge, 2)
