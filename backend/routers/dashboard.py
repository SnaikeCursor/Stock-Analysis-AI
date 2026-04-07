"""Dashboard router — GET /api/dashboard.

Aggregates current portfolio, next rebalancing countdown, performance since entry,
and recent alerts into a single overview payload.

Lag60-SA: semi-annual rebalancing (January and July).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db import get_session

router = APIRouter(prefix="/api", tags=["dashboard"])


def _next_rebalancing_date(today: date | None = None) -> date:
    """Return the next semi-annual rebalancing anchor (1 Jan or 1 Jul).

    Lag60-SA uses semi-annual rebalancing — the next anchor is whichever
    of 1 Jan or 1 Jul (or next year's 1 Jan) comes first after today.
    """
    today = today or date.today()
    candidates = [
        date(today.year, 1, 1),
        date(today.year, 7, 1),
        date(today.year + 1, 1, 1),
    ]
    return min(d for d in candidates if d > today)


@router.get("/dashboard")
async def get_dashboard(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    portfolio_svc = request.app.state.portfolio_service

    # --- Current portfolio & P&L ---
    signal = await portfolio_svc.get_latest_signal(session)
    positions: list[dict[str, Any]] = []
    if signal is not None:
        positions = await portfolio_svc.compute_live_pnl(session)

    # --- Rebalancing countdown (semi-annual) ---
    next_rebal = _next_rebalancing_date()
    days_until = (next_rebal - date.today()).days

    # --- Latest signal metadata ---
    signal_meta: dict[str, Any] | None = None
    if signal is not None:
        signal_meta = {
            "id": signal.id,
            "cutoff_date": signal.cutoff_date,
            "created_at": signal.created_at.isoformat() if signal.created_at else None,
            "status": signal.status.value if hasattr(signal.status, "value") else signal.status,
        }

    # --- Recent alerts ---
    all_alerts = await portfolio_svc.get_unread_alerts(session)
    recent_alerts = [
        {
            "id": a.id,
            "type": a.type.value if hasattr(a.type, "value") else a.type,
            "message": a.message,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in all_alerts[:5]
    ]

    return {
        "signal": signal_meta,
        "positions": positions,
        "next_rebalancing": {
            "date": next_rebal.isoformat(),
            "days_until": days_until,
        },
        "alerts": recent_alerts,
        "model_info": {
            "phase": "Lag60-SA",
            "rebalance_freq": "semi-annual",
            "description": "Regression (CS z-score) + PIT fundamentals (60d lag), semi-annual rebalancing",
        },
    }
