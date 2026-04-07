"""Signals router — signal generation and retrieval.

Endpoints:
  POST /api/signals/generate   — generate a new trading signal
  GET  /api/signals/latest     — most recent active signal
  GET  /api/signals/history    — all past signals
"""

from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db import SignalStatus, get_session

router = APIRouter(prefix="/api/signals", tags=["signals"])


# ------------------------------------------------------------------
# Request / response schemas
# ------------------------------------------------------------------


class GenerateSignalRequest(BaseModel):
    cutoff_date: str | None = Field(
        None,
        description="ISO date for feature computation. Defaults to today.",
    )
    top_n: int = Field(5, ge=1, le=20)
    max_weight: float = Field(0.30, ge=0.05, le=1.0)


class PositionOut(BaseModel):
    ticker: str
    weight: float
    predicted_return: float = Field(
        description="Predicted cross-sectional z-scored forward return.",
    )
    current_price: float | None = Field(
        None,
        description="Latest close from cached OHLCV (set on generate for share sizing).",
    )


class SignalOut(BaseModel):
    id: int
    cutoff_date: str
    portfolio: list[PositionOut]
    status: str
    created_at: str | None = None
    model_phase: str = "Lag60-SA"
    rebalance_freq: str = "semi-annual"


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/generate")
async def generate_signal(
    body: GenerateSignalRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> SignalOut:
    model_svc = request.app.state.model_service
    portfolio_svc = request.app.state.portfolio_service

    cutoff = body.cutoff_date or date.today().isoformat()

    try:
        result = model_svc.generate_signal(
            cutoff,
            top_n=body.top_n,
            max_weight=body.max_weight,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {exc}")

    signal = await portfolio_svc.save_signal(
        session,
        cutoff_date=result.cutoff_date,
        regime_label=result.regime_label,
        regime_confidence=result.regime_confidence,
        portfolio_json=result.portfolio_json,
        portfolio=result.portfolio,
        status=SignalStatus.PENDING,
    )

    tickers = [p["ticker"] for p in result.portfolio]
    price_map = portfolio_svc.get_current_prices_for_tickers(tickers)

    return SignalOut(
        id=signal.id,
        cutoff_date=signal.cutoff_date,
        portfolio=[
            PositionOut(
                ticker=p["ticker"],
                weight=p["weight"],
                predicted_return=p.get("predicted_return", 0.0),
                current_price=price_map.get(p["ticker"]),
            )
            for p in result.portfolio
        ],
        status=signal.status.value if hasattr(signal.status, "value") else signal.status,
        created_at=signal.created_at.isoformat() if signal.created_at else None,
    )


@router.get("/latest")
async def get_latest_signal(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> SignalOut | None:
    portfolio_svc = request.app.state.portfolio_service
    signal = await portfolio_svc.get_latest_signal(session)

    if signal is None:
        return None

    portfolio = json.loads(signal.portfolio_json) if signal.portfolio_json else []

    return SignalOut(
        id=signal.id,
        cutoff_date=signal.cutoff_date,
        portfolio=[
            PositionOut(
                ticker=p["ticker"],
                weight=p["weight"],
                predicted_return=p.get("predicted_return", p.get("p_winner", 0.0)),
            )
            for p in portfolio
        ],
        status=signal.status.value if hasattr(signal.status, "value") else signal.status,
        created_at=signal.created_at.isoformat() if signal.created_at else None,
    )


@router.get("/history")
async def get_signal_history(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[SignalOut]:
    portfolio_svc = request.app.state.portfolio_service
    signals = await portfolio_svc.get_signal_history(session)

    results: list[SignalOut] = []
    for sig in signals:
        portfolio = json.loads(sig.portfolio_json) if sig.portfolio_json else []
        results.append(
            SignalOut(
                id=sig.id,
                cutoff_date=sig.cutoff_date,
                portfolio=[
                    PositionOut(
                        ticker=p["ticker"],
                        weight=p["weight"],
                        predicted_return=p.get("predicted_return", p.get("p_winner", 0.0)),
                    )
                    for p in portfolio
                ],
                status=sig.status.value if hasattr(sig.status, "value") else sig.status,
                created_at=sig.created_at.isoformat() if sig.created_at else None,
            )
        )

    return results
