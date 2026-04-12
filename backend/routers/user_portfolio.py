"""Per-user manual portfolio (UUID via X-User-ID) — cash, positions, Swissquote fees."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import get_or_create_user
from backend.models.db import UserPosition, UserProfile, get_session
from backend.services.fee_calculator import swissquote_fee
from backend.services.user_portfolio_service import UserPortfolioService

router = APIRouter(prefix="/api/me", tags=["user-portfolio"])


def _user_portfolio_service(request: Request) -> UserPortfolioService:
    return UserPortfolioService(request.app.state.data_service)


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


class DepositBody(BaseModel):
    amount: float = Field(..., gt=0, description="CHF to add to cash balance")


class AddPositionBody(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    shares: int = Field(..., ge=1)
    entry_price: float = Field(..., gt=0)
    entry_date: str = Field(..., min_length=10, max_length=10, description="ISO date YYYY-MM-DD")


class ClosePositionBody(BaseModel):
    exit_price: float = Field(..., gt=0)
    exit_date: str | None = Field(None, description="ISO date; defaults to today")


class ApplySignalBody(BaseModel):
    signal_id: int = Field(..., description="ID of the signal to apply")
    investment_amount: float = Field(..., gt=0, description="CHF to invest from cash balance")


class WithdrawBody(BaseModel):
    amount: float = Field(..., gt=0, description="CHF to withdraw from cash balance")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@router.get("/portfolio")
async def get_my_portfolio(
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    return await svc.get_overview(session, user)


@router.get("/portfolio/summary")
async def get_my_portfolio_summary(
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    return await svc.get_summary(session, user)


@router.post("/portfolio/deposit")
async def deposit_cash(
    body: DepositBody,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    try:
        pf = await svc.deposit(session, user, body.amount)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"cash_balance": round(float(pf.cash_balance), 2)}


@router.post("/portfolio/positions")
async def add_position(
    body: AddPositionBody,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    try:
        pos = await svc.add_position(
            session,
            user,
            ticker=body.ticker,
            shares=body.shares,
            entry_price=body.entry_price,
            entry_date=body.entry_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _position_to_dict(pos)


@router.put("/portfolio/positions/{position_id}/close")
async def close_position(
    position_id: int,
    body: ClosePositionBody,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    try:
        pos = await svc.close_position(
            session,
            user,
            position_id,
            exit_price=body.exit_price,
            exit_date=body.exit_date,
        )
    except ValueError as exc:
        msg = str(exc)
        code = 404 if "not found" in msg.lower() else 400
        raise HTTPException(status_code=code, detail=msg) from exc
    return _position_to_dict(pos)


@router.delete("/portfolio/positions/{position_id}", status_code=204)
async def delete_position(
    position_id: int,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> None:
    try:
        await svc.delete_position(session, user, position_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/portfolio/performance")
async def get_performance(
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    """Portfolio performance with Modified Dietz return (cash-flow adjusted)."""
    return await svc.compute_performance(session, user)


@router.post("/portfolio/apply-signal")
async def apply_signal(
    body: ApplySignalBody,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    """Buy signal-recommended positions into the user portfolio."""
    try:
        result = await svc.apply_signal(
            session,
            user,
            signal_id=body.signal_id,
            investment_amount=body.investment_amount,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/portfolio/withdraw")
async def withdraw_cash(
    body: WithdrawBody,
    user: UserProfile = Depends(get_or_create_user),
    session: AsyncSession = Depends(get_session),
    svc: UserPortfolioService = Depends(_user_portfolio_service),
) -> dict[str, Any]:
    try:
        pf = await svc.withdraw(session, user, body.amount)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"cash_balance": round(float(pf.cash_balance), 2)}


@router.get("/swissquote-fee")
async def estimate_swissquote_fee(volume_chf: float) -> dict[str, float]:
    """Return estimated total fee (CHF) for a given notional trade size."""
    if volume_chf <= 0:
        raise HTTPException(status_code=400, detail="volume_chf must be positive")
    return {"volume_chf": volume_chf, "fee_chf": swissquote_fee(volume_chf)}


def _position_to_dict(pos: UserPosition) -> dict[str, Any]:
    return {
        "id": pos.id,
        "ticker": pos.ticker,
        "shares": pos.shares,
        "entry_price": pos.entry_price,
        "entry_date": pos.entry_date,
        "entry_total": pos.entry_total,
        "entry_fee": pos.entry_fee,
        "exit_price": pos.exit_price,
        "exit_date": pos.exit_date,
        "exit_total": pos.exit_total,
        "exit_fee": pos.exit_fee,
        "pnl_abs": pos.pnl_abs,
        "pnl_pct": pos.pnl_pct,
        "status": pos.status,
    }
