"""Alerts router — notification management.

Endpoints:
  GET /api/alerts          — list all alerts (optionally filter by unread)
  PUT /api/alerts/{id}/read — mark a single alert as read
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db import get_session

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


class AlertOut(BaseModel):
    id: int
    type: str
    message: str
    created_at: str | None = None
    read: bool


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("")
async def list_alerts(
    request: Request,
    unread_only: bool = Query(False, description="Return only unread alerts"),
    session: AsyncSession = Depends(get_session),
) -> list[AlertOut]:
    portfolio_svc = request.app.state.portfolio_service

    if unread_only:
        alerts = await portfolio_svc.get_unread_alerts(session)
    else:
        alerts = await portfolio_svc.get_all_alerts(session)

    return [
        AlertOut(
            id=a.id,
            type=a.type.value if hasattr(a.type, "value") else a.type,
            message=a.message,
            created_at=a.created_at.isoformat() if a.created_at else None,
            read=a.read,
        )
        for a in alerts
    ]


@router.put("/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> AlertOut:
    portfolio_svc = request.app.state.portfolio_service

    try:
        alert = await portfolio_svc.mark_alert_read(session, alert_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Alert not found: {exc}")

    return AlertOut(
        id=alert.id,
        type=alert.type.value if hasattr(alert.type, "value") else alert.type,
        message=alert.message,
        created_at=alert.created_at.isoformat() if alert.created_at else None,
        read=alert.read,
    )
