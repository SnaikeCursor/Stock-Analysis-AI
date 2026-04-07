"""APScheduler integration — weekly OHLCV refresh, semi-annual signal.

Lag60-SA: semi-annual rebalancing (January + July) with 60-day publication lag.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.models.db import SignalStatus, async_session_factory
from backend.services.data_service import DataService
from backend.services.model_service import ModelService
from backend.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


def _scheduler_enabled() -> bool:
    return os.environ.get("ENABLE_SCHEDULER", "true").lower() in ("1", "true", "yes")


def _last_smi_trading_date_str(data: DataService) -> str:
    """Latest trading date available in cached ^SSMI OHLCV (ISO)."""
    smi = data.get_smi_ohlcv()
    if smi.empty:
        raise ValueError("SMI OHLCV is empty")
    last_ts = pd.Timestamp(smi.index.max())
    return last_ts.strftime("%Y-%m-%d")


def create_scheduler(
    data_service: DataService,
    model_service: ModelService,
    portfolio_service: PortfolioService,
) -> AsyncIOScheduler:
    """Build an :class:`AsyncIOScheduler` with OHLCV and signal jobs.

    Call ``scheduler.start()`` after the asyncio event loop is running
    (e.g. inside FastAPI lifespan). Jobs are no-ops when the scheduler is
    disabled via ``ENABLE_SCHEDULER=false``.

    Lag60-SA: Signal generation runs semi-annually (January 2nd + July 1st),
    matching the semi-annual rebalancing frequency.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    async def job_weekly_ohlcv() -> None:
        if not _scheduler_enabled():
            return
        logger.info("Scheduler: weekly OHLCV refresh starting")
        try:

            def _refresh() -> int:
                return data_service.refresh_ohlcv()

            n = await asyncio.to_thread(_refresh)
            logger.info("Scheduler: OHLCV refresh finished (%d tickers)", n)
        except Exception:
            logger.exception("Scheduler: weekly OHLCV refresh failed")

    async def job_semi_annual_signal() -> None:
        """Generate semi-annual trading signal (Lag60-SA, January + July)."""
        if not _scheduler_enabled():
            return
        if not model_service.is_loaded:
            try:
                await asyncio.to_thread(model_service.load_model)
            except FileNotFoundError:
                logger.warning(
                    "Scheduler: semi-annual signal skipped — model cache missing"
                )
                return

        logger.info("Scheduler: semi-annual signal generation starting (Lag60-SA)")
        try:
            cutoff = await asyncio.to_thread(_last_smi_trading_date_str, data_service)

            def _gen():
                return model_service.generate_signal(cutoff)

            result = await asyncio.to_thread(_gen)

            async with async_session_factory() as session:
                signal = await portfolio_service.save_signal(
                    session,
                    cutoff_date=result.cutoff_date,
                    regime_label=result.regime_label,
                    regime_confidence=result.regime_confidence,
                    portfolio_json=result.portfolio_json,
                    portfolio=result.portfolio,
                    status=SignalStatus.PENDING,
                )
                new_signal_id = signal.id

            async with async_session_factory() as session:
                await portfolio_service.create_rebalance_proposal_for_new_signal(
                    session,
                    new_signal_id=new_signal_id,
                )

            logger.info(
                "Scheduler: semi-annual signal saved as PENDING (id=%s, cutoff=%s)",
                new_signal_id,
                result.cutoff_date,
            )
        except Exception:
            logger.exception("Scheduler: semi-annual signal generation failed")

    # Weekly Monday 05:30 UTC — OHLCV refresh
    scheduler.add_job(
        job_weekly_ohlcv,
        CronTrigger(day_of_week="mon", hour=5, minute=30, timezone="UTC"),
        id="weekly_ohlcv_refresh",
        replace_existing=True,
    )

    # Jan 2nd 06:00 UTC — semi-annual PENDING signal (H1)
    scheduler.add_job(
        job_semi_annual_signal,
        CronTrigger(month=1, day=2, hour=6, minute=0, timezone="UTC"),
        id="semi_annual_signal_jan",
        replace_existing=True,
    )

    # Jul 1st 06:00 UTC — semi-annual PENDING signal (H2)
    scheduler.add_job(
        job_semi_annual_signal,
        CronTrigger(month=7, day=1, hour=6, minute=0, timezone="UTC"),
        id="semi_annual_signal_jul",
        replace_existing=True,
    )

    return scheduler
