"""APScheduler integration — daily OHLCV refresh, semi-annual Eulerpool + signal jobs.

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

    async def job_daily_ohlcv() -> None:
        if not _scheduler_enabled():
            return
        logger.info("Scheduler: daily OHLCV refresh starting")
        try:

            def _refresh() -> int:
                return data_service.refresh_ohlcv()

            n = await asyncio.to_thread(_refresh)
            logger.info("Scheduler: OHLCV refresh finished (%d tickers)", n)
        except Exception:
            logger.exception("Scheduler: daily OHLCV refresh failed")

    async def job_semi_annual_eulerpool_refresh() -> None:
        """Force-refresh Eulerpool PIT fundamentals before semi-annual rebalances."""
        if not _scheduler_enabled():
            return

        logger.info("Scheduler: semi-annual Eulerpool refresh starting")
        try:
            # Ensure OHLCV/universe cache is available before resolving liquid tickers.
            await asyncio.to_thread(data_service.load_cached)
            stock_tickers = await asyncio.to_thread(data_service.get_liquid_tickers)

            def _refresh_eulerpool() -> tuple[dict[str, list[dict]], dict[str, dict]]:
                from src.eulerpool_fundamentals import fetch_all_quarterly

                return fetch_all_quarterly(stock_tickers, force=True)

            eulerpool_q, eulerpool_p = await asyncio.to_thread(_refresh_eulerpool)
            data_service._eulerpool_quarterly = eulerpool_q
            data_service._eulerpool_profiles = eulerpool_p

            n_with_data = sum(1 for v in eulerpool_q.values() if v)
            logger.info(
                "Scheduler: semi-annual Eulerpool refresh finished (%d/%d tickers with quarterly data)",
                n_with_data,
                len(stock_tickers),
            )
        except Exception:
            logger.exception("Scheduler: semi-annual Eulerpool refresh failed")

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

            try:
                from backend.services.notification_service import notify_new_signal

                top_picks = [p["ticker"] for p in result.portfolio]
                await asyncio.to_thread(
                    notify_new_signal,
                    signal_id=new_signal_id,
                    cutoff_date=result.cutoff_date,
                    n_positions=len(result.portfolio),
                    regime=result.regime_label,
                    top_picks=top_picks,
                )
            except Exception:
                logger.exception("Scheduler: email notification failed (non-fatal)")
        except Exception:
            logger.exception("Scheduler: semi-annual signal generation failed")

    # Daily weekday 05:30 UTC — OHLCV refresh
    scheduler.add_job(
        job_daily_ohlcv,
        CronTrigger(day_of_week="mon-fri", hour=5, minute=30, timezone="UTC"),
        id="daily_ohlcv_refresh",
        replace_existing=True,
    )

    # Dec 28th 04:00 UTC — force-refresh Eulerpool data before Jan rebalance
    scheduler.add_job(
        job_semi_annual_eulerpool_refresh,
        CronTrigger(month=12, day=28, hour=4, minute=0, timezone="UTC"),
        id="semi_annual_eulerpool_refresh_dec",
        replace_existing=True,
    )

    # Jun 28th 04:00 UTC — force-refresh Eulerpool data before Jul rebalance
    scheduler.add_job(
        job_semi_annual_eulerpool_refresh,
        CronTrigger(month=6, day=28, hour=4, minute=0, timezone="UTC"),
        id="semi_annual_eulerpool_refresh_jun",
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
