"""FastAPI application — entry point for the Stock Analysis AI webapp.

Configures CORS, wires up the service layer as singletons, initialises the
database on startup, and mounts all API routers under ``/api``.

Run with::

    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.db import init_db
from backend.routers import alerts, dashboard, history, portfolio, signals, user_portfolio
from backend.services.data_service import DataService
from backend.services.model_service import ModelService
from backend.services.portfolio_service import PortfolioService
from backend.services.scheduler import create_scheduler

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Singleton services (shared across requests)
# ------------------------------------------------------------------

data_service = DataService()
model_service = ModelService(data_service)
portfolio_service = PortfolioService(data_service)

_apscheduler = create_scheduler(data_service, model_service, portfolio_service)


# ------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting Stock Analysis AI backend …")

    await init_db()
    logger.info("Database initialised")

    _apscheduler.start()
    logger.info(
        "APScheduler started — weekly OHLCV (Mon 05:30 UTC), "
        "semi-annual signals (2 Jan & 1 Jul 06:00 UTC)"
    )

    async def _bootstrap_heavy_io() -> None:
        """OHLCV + model load can take many minutes on a cold cache (yfinance).

        Must not run before ``yield`` — Uvicorn only accepts connections after
        startup completes, so Railway/Docker healthchecks would time out.
        """
        try:
            await asyncio.to_thread(data_service.load_cached)
            logger.info("OHLCV cache loaded (%d tickers)", len(data_service.ohlcv))
        except Exception as exc:
            logger.warning("Could not load OHLCV cache on startup: %s", exc)

        try:
            await asyncio.to_thread(model_service.load_model)
            logger.info("Lag60 Semi-Annual regression model loaded")
        except Exception as exc:
            logger.warning(
                "Model not available at startup — signal generation will fail until trained: %s",
                exc,
            )

    bootstrap_task = asyncio.create_task(_bootstrap_heavy_io())

    yield

    bootstrap_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await bootstrap_task

    _apscheduler.shutdown(wait=False)
    logger.info("APScheduler stopped; shutting down …")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="Stock Analysis AI",
    description="Swiss SPI stock selection — Lag-60 Semi-Annual (regression + PIT fundamentals, semi-annual rebalancing)",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Dependency injection helpers (used by routers via app.state)
# ------------------------------------------------------------------

app.state.data_service = data_service
app.state.model_service = model_service
app.state.portfolio_service = portfolio_service

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------

app.include_router(dashboard.router)
app.include_router(signals.router)
app.include_router(history.router)
app.include_router(portfolio.router)
app.include_router(user_portfolio.router)
app.include_router(alerts.router)


@app.get("/api/health")
async def health():
    data_ok = data_service.is_loaded
    model_ok = model_service.is_loaded
    if data_ok and model_ok:
        status = "ok"
    elif data_ok:
        status = "degraded"
    else:
        status = "unavailable"
    return {
        "status": status,
        "data_loaded": data_ok,
        "model_loaded": model_ok,
    }
