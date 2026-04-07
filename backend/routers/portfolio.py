"""Portfolio router — position management, live P&L, and rebalancing.

Endpoints:
  POST /api/portfolio/activate           — activate a PENDING signal with shares + entry prices
  GET  /api/portfolio                    — active positions with weights
  GET  /api/portfolio/pnl                — live P&L for active positions
  PUT  /api/portfolio/positions/{id}     — edit shares, prices, dates (recalculates P&L)
  PUT  /api/portfolio/positions/{id}/close — close a position with exit price
  GET  /api/portfolio/history            — closed/historical positions
  GET  /api/portfolio/rebalance-proposal — pending rebalance proposal (if any)
  POST /api/portfolio/execute-rebalance  — confirm and execute rebalancing swaps
  POST /api/portfolio/dismiss-rebalance  — dismiss a pending rebalance proposal
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select

from backend.models.db import RebalanceProposal, get_session

router = APIRouter(prefix="/api", tags=["portfolio"])


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------


class ActivatePositionEntry(BaseModel):
    ticker: str
    shares: int = Field(..., ge=1)
    entry_price: float = Field(..., gt=0)
    entry_date: str | None = Field(None, description="ISO date; defaults to today")


class ActivatePortfolioRequest(BaseModel):
    signal_id: int
    investment_amount: float = Field(..., gt=0, description="Total CHF invested")
    positions: list[ActivatePositionEntry] = Field(
        ..., min_length=1, description="User-confirmed trade data per ticker"
    )


class ActivatePortfolioResponse(BaseModel):
    signal_id: int
    status: str
    positions_activated: int
    investment_amount: float


class ClosePositionRequest(BaseModel):
    exit_price: float = Field(..., gt=0)
    exit_date: str | None = Field(None, description="ISO date; defaults to today")


class UpdatePositionRequest(BaseModel):
    """Partial update: only sent fields are applied. Use null to clear optional values."""

    model_config = ConfigDict(extra="forbid")

    shares: int | None = Field(None, ge=1, description="Number of shares")
    entry_price: float | None = Field(None, gt=0)
    entry_date: str | None = Field(None, description="ISO date YYYY-MM-DD")
    exit_price: float | None = Field(None, gt=0)
    exit_date: str | None = Field(None, description="ISO date YYYY-MM-DD")


class ExecutedTradeEntry(BaseModel):
    ticker: str
    action: str = Field(..., description="buy, sell, or hold")
    shares: int = Field(..., ge=0)
    price: float = Field(..., gt=0)
    date: str | None = Field(None, description="ISO date; defaults to today")


class ExecuteRebalanceRequest(BaseModel):
    rebalance_id: int
    executed_trades: list[ExecutedTradeEntry] = Field(
        ..., min_length=1, description="User-confirmed trade data per ticker"
    )


class ExecuteRebalanceResponse(BaseModel):
    signal_id: int
    status: str
    trades_processed: int
    old_signal_id: int
    new_signal_id: int


class DismissRebalanceRequest(BaseModel):
    rebalance_id: int


class RebalanceInstructionOut(BaseModel):
    ticker: str
    action: str
    shares: int
    estimated_price: float | None = None
    estimated_value: float | None = None
    reason: str | None = None


class RebalanceProposalOut(BaseModel):
    id: int
    old_signal_id: int
    new_signal_id: int
    status: str
    created_at: str | None = None
    portfolio_value: float | None = None
    swap_count: int | None = None
    instructions: list[RebalanceInstructionOut] = []
    note: str | None = None


class PositionDetail(BaseModel):
    id: int
    ticker: str
    weight: float
    shares: int | None = None
    entry_price: float | None = None
    entry_date: str | None = None
    entry_total: float | None = None
    exit_price: float | None = None
    exit_date: str | None = None
    exit_total: float | None = None
    pnl_pct: float | None = None


class PnlEntry(BaseModel):
    id: int
    ticker: str
    weight: float
    shares: int | None = None
    entry_price: float | None = None
    entry_date: str | None = None
    entry_total: float | None = None
    current_price: float | None = None
    current_value: float | None = None
    pnl_pct: float | None = None
    pnl_abs: float | None = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/portfolio/activate")
async def activate_portfolio(
    body: ActivatePortfolioRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ActivatePortfolioResponse:
    """Activate a PENDING signal: close old positions at market, stamp new ones with shares."""
    portfolio_svc = request.app.state.portfolio_service

    pos_dicts = [p.model_dump(mode="python") for p in body.positions]

    try:
        signal = await portfolio_svc.activate_portfolio(
            session,
            signal_id=body.signal_id,
            investment_amount=body.investment_amount,
            positions=pos_dicts,
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc

    status = signal.status.value if hasattr(signal.status, "value") else signal.status
    return ActivatePortfolioResponse(
        signal_id=signal.id,
        status=status,
        positions_activated=len(body.positions),
        investment_amount=body.investment_amount,
    )


@router.get("/portfolio")
async def get_active_portfolio(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[PositionDetail]:
    portfolio_svc = request.app.state.portfolio_service
    positions = await portfolio_svc.get_active_positions(session)

    return [
        PositionDetail(
            id=p.id,
            ticker=p.ticker,
            weight=p.weight,
            shares=p.shares,
            entry_price=p.entry_price,
            entry_date=p.entry_date,
            entry_total=p.entry_total,
            exit_price=p.exit_price,
            exit_date=p.exit_date,
            exit_total=p.exit_total,
            pnl_pct=p.pnl_pct,
        )
        for p in positions
    ]


@router.get("/portfolio/pnl")
async def get_live_pnl(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[PnlEntry]:
    portfolio_svc = request.app.state.portfolio_service
    pnl_data = await portfolio_svc.compute_live_pnl(session)

    return [PnlEntry(**entry) for entry in pnl_data]


@router.put("/portfolio/positions/{position_id}/close")
async def close_position(
    position_id: int,
    body: ClosePositionRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> PositionDetail:
    portfolio_svc = request.app.state.portfolio_service
    exit_dt = body.exit_date or date.today().isoformat()

    try:
        pos = await portfolio_svc.close_position(
            session,
            position_id,
            body.exit_price,
            exit_dt,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Position not found: {exc}")

    return PositionDetail(
        id=pos.id,
        ticker=pos.ticker,
        weight=pos.weight,
        shares=pos.shares,
        entry_price=pos.entry_price,
        entry_date=pos.entry_date,
        entry_total=pos.entry_total,
        exit_price=pos.exit_price,
        exit_date=pos.exit_date,
        exit_total=pos.exit_total,
        pnl_pct=pos.pnl_pct,
    )


@router.put("/portfolio/positions/{position_id}")
async def update_position(
    position_id: int,
    body: UpdatePositionRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> PositionDetail:
    """Update shares, prices, and dates; entry/exit totals and P&L are recalculated."""
    if not body.model_fields_set:
        raise HTTPException(
            status_code=400,
            detail="At least one field must be provided",
        )

    portfolio_svc = request.app.state.portfolio_service
    patch = body.model_dump(mode="python", exclude_unset=True)

    try:
        pos = await portfolio_svc.update_position(session, position_id, patch)
    except ValueError as exc:
        msg = str(exc)
        if "No position with id" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc

    return PositionDetail(
        id=pos.id,
        ticker=pos.ticker,
        weight=pos.weight,
        shares=pos.shares,
        entry_price=pos.entry_price,
        entry_date=pos.entry_date,
        entry_total=pos.entry_total,
        exit_price=pos.exit_price,
        exit_date=pos.exit_date,
        exit_total=pos.exit_total,
        pnl_pct=pos.pnl_pct,
    )


@router.get("/portfolio/rebalance-proposal")
async def get_rebalance_proposal(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> RebalanceProposalOut | None:
    """Return the most recent pending rebalance proposal, or null."""
    portfolio_svc = request.app.state.portfolio_service
    proposal = await portfolio_svc.get_pending_rebalance_proposal(session)
    if proposal is None:
        return None

    try:
        payload = json.loads(proposal.instructions_json)
    except (json.JSONDecodeError, TypeError):
        payload = {}

    raw_instructions = payload.get("instructions", [])
    instructions = [
        RebalanceInstructionOut(
            ticker=i.get("ticker", ""),
            action=i.get("action", ""),
            shares=int(i.get("shares", 0)),
            estimated_price=i.get("estimated_price"),
            estimated_value=i.get("estimated_value"),
            reason=i.get("reason"),
        )
        for i in raw_instructions
    ]

    status = (
        proposal.status.value
        if hasattr(proposal.status, "value")
        else proposal.status
    )
    return RebalanceProposalOut(
        id=proposal.id,
        old_signal_id=proposal.old_signal_id,
        new_signal_id=proposal.new_signal_id,
        status=status,
        created_at=(
            proposal.created_at.isoformat() if proposal.created_at else None
        ),
        portfolio_value=payload.get("portfolio_value"),
        swap_count=payload.get("swap_count"),
        instructions=instructions,
        note=payload.get("note"),
    )


@router.post("/portfolio/execute-rebalance")
async def execute_rebalance(
    body: ExecuteRebalanceRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ExecuteRebalanceResponse:
    """Confirm and execute a pending rebalance: close old, open new positions."""
    portfolio_svc = request.app.state.portfolio_service

    trade_dicts = [t.model_dump(mode="python") for t in body.executed_trades]

    try:
        signal = await portfolio_svc.execute_rebalance(
            session,
            rebalance_id=body.rebalance_id,
            executed_trades=trade_dicts,
        )
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg:
            raise HTTPException(status_code=404, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc

    # Retrieve proposal to include IDs in response
    stmt = (
        select(RebalanceProposal)
        .where(RebalanceProposal.id == body.rebalance_id)
    )
    proposal = (await session.execute(stmt)).scalar_one()

    status = signal.status.value if hasattr(signal.status, "value") else signal.status
    return ExecuteRebalanceResponse(
        signal_id=signal.id,
        status=status,
        trades_processed=len(body.executed_trades),
        old_signal_id=proposal.old_signal_id,
        new_signal_id=proposal.new_signal_id,
    )


@router.post("/portfolio/dismiss-rebalance")
async def dismiss_rebalance(
    body: DismissRebalanceRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Dismiss a pending rebalance proposal."""
    portfolio_svc = request.app.state.portfolio_service

    try:
        proposal = await portfolio_svc.dismiss_rebalance_proposal(
            session, body.rebalance_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    status = (
        proposal.status.value
        if hasattr(proposal.status, "value")
        else proposal.status
    )
    return {
        "id": proposal.id,
        "status": status,
        "message": f"Proposal {proposal.id} dismissed",
    }


@router.get("/portfolio/history")
async def get_portfolio_history(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """Return all signals with their associated positions (newest first)."""
    portfolio_svc = request.app.state.portfolio_service
    signals = await portfolio_svc.get_signal_history(session)

    results: list[dict[str, Any]] = []
    for sig in signals:
        results.append(
            {
                "signal_id": sig.id,
                "cutoff_date": sig.cutoff_date,
                "regime_label": sig.regime_label,
                "status": sig.status.value if hasattr(sig.status, "value") else sig.status,
                "created_at": sig.created_at.isoformat() if sig.created_at else None,
                "positions": [
                    {
                        "id": p.id,
                        "ticker": p.ticker,
                        "weight": p.weight,
                        "shares": p.shares,
                        "entry_price": p.entry_price,
                        "entry_date": p.entry_date,
                        "entry_total": p.entry_total,
                        "exit_price": p.exit_price,
                        "exit_date": p.exit_date,
                        "exit_total": p.exit_total,
                        "pnl_pct": p.pnl_pct,
                    }
                    for p in sig.positions
                ],
            }
        )

    return results
