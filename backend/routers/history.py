"""History router — historical backtest performance.

Endpoints:
  GET /api/history/performance — per-year long-only vs benchmark summary
  GET /api/history/quarterly   — granular quarterly detail (returns, turnover, costs)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/history", tags=["history"])


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
