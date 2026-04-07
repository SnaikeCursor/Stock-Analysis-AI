"""History router — historical backtest performance.

Endpoints:
  GET  /api/history/performance — per-year long-only vs benchmark summary
  GET  /api/history/quarterly   — granular quarterly detail (returns, turnover, costs)
  POST /api/history/simulate    — capital-space backtest with transaction log
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/history", tags=["history"])


class SimulateRequest(BaseModel):
    """Input for the backtest simulator."""

    start_date: str = Field(
        default="2015-01-01",
        description="ISO date — simulation starts from this year (min 2015).",
    )
    initial_capital: float = Field(
        default=100_000,
        ge=1_000,
        le=100_000_000,
        description="Starting capital in CHF.",
    )
    costs_bps: float = Field(
        default=40.0,
        ge=0,
        le=500,
        description="One-way transaction costs in basis points.",
    )


@router.get("/performance")
async def get_yearly_performance(
    request: Request,
    start_year: int = Query(2015, ge=2012, le=2030),
    end_year: int = Query(2025, ge=2012, le=2030),
) -> dict[str, Any]:
    """Run (or return cached) walk-forward quarterly backtest and return
    per-year performance metrics: long-only return, benchmark return, and costs.
    """
    model_svc = request.app.state.model_service

    oos_years = list(range(start_year, end_year + 1))

    try:
        result = model_svc.run_historical_backtest(oos_years=oos_years)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    per_year_serialised: dict[str, Any] = {}
    for year, metrics in result["per_year"].items():
        per_year_serialised[str(year)] = metrics

    return {
        "per_year": per_year_serialised,
        "total_costs_bps": result["total_costs_bps"],
        "rebalance_freq": result["rebalance_freq"],
    }


@router.get("/quarterly")
async def get_quarterly_detail(
    request: Request,
    start_year: int = Query(2015, ge=2012, le=2030),
    end_year: int = Query(2025, ge=2012, le=2030),
) -> dict[str, Any]:
    """Return quarterly-granularity data: per-quarter returns, turnover, costs."""
    model_svc = request.app.state.model_service

    oos_years = list(range(start_year, end_year + 1))

    try:
        result = model_svc.run_historical_backtest(oos_years=oos_years)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}")

    return {
        "quarterly_detail": result["quarterly_detail"],
        "total_costs_bps": result["total_costs_bps"],
    }


@router.post("/simulate")
async def simulate_backtest(
    request: Request,
    body: SimulateRequest,
) -> dict[str, Any]:
    """Run a capital-space backtest simulation.

    Returns a daily portfolio-value timeline, a list of every buy/sell
    transaction, and summary KPIs (total return, CAGR, Sharpe, max
    drawdown, total costs).
    """
    model_svc = request.app.state.model_service

    try:
        result = model_svc.run_backtest_simulation(
            start_date=body.start_date,
            initial_capital=body.initial_capital,
            costs_bps=body.costs_bps,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Backtest simulation failed")
        raise HTTPException(
            status_code=500, detail=f"Simulation failed: {exc}",
        )

    return result
