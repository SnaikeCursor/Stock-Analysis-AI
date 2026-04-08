"""User-owned manual portfolio: cash, positions, Swissquote-style fees, live P&L."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models.db import CashTransaction, UserPortfolio, UserPosition, UserProfile
from backend.services.data_service import DataService
from backend.services.fee_calculator import swissquote_fee

logger = logging.getLogger(__name__)


async def _get_or_create_portfolio(
    session: AsyncSession,
    user: UserProfile,
    *,
    load_positions: bool = False,
) -> UserPortfolio:
    opts = [selectinload(UserPortfolio.positions)] if load_positions else []
    stmt = select(UserPortfolio).where(UserPortfolio.user_id == user.id).options(*opts)
    res = await session.execute(stmt)
    pf = res.scalar_one_or_none()
    if pf is None:
        pf = UserPortfolio(user_id=user.id, cash_balance=0.0)
        session.add(pf)
        await session.commit()
        if load_positions:
            stmt2 = (
                select(UserPortfolio)
                .where(UserPortfolio.id == pf.id)
                .options(selectinload(UserPortfolio.positions))
            )
            res2 = await session.execute(stmt2)
            pf = res2.scalar_one()
        else:
            await session.refresh(pf)
    return pf


def _current_price_for_ticker(data: DataService, ticker: str) -> float | None:
    try:
        df = data.get_ticker_price(ticker)
        if df is not None and not df.empty and "Close" in df.columns:
            return round(float(df["Close"].iloc[-1]), 4)
    except (KeyError, Exception) as exc:
        logger.debug("Price lookup failed for %s: %s", ticker, exc)
    return None


class UserPortfolioService:
    def __init__(self, data_service: DataService) -> None:
        self._data = data_service

    async def get_overview(
        self,
        session: AsyncSession,
        user: UserProfile,
    ) -> dict[str, Any]:
        pf = await _get_or_create_portfolio(session, user, load_positions=True)

        open_rows: list[dict[str, Any]] = []
        closed_rows: list[dict[str, Any]] = []

        tickers = [p.ticker for p in pf.positions if p.status == "open"]
        prices: dict[str, float | None] = {}
        if tickers:
            prices = await asyncio.to_thread(self._prices_map, tickers)

        for pos in sorted(pf.positions, key=lambda x: x.id):
            if pos.status == "open":
                cur = prices.get(pos.ticker)
                pnl_pct = None
                pnl_abs = None
                cur_val = None
                if cur is not None and pos.entry_price and pos.entry_price > 0:
                    pnl_pct = round((cur - float(pos.entry_price)) / float(pos.entry_price), 6)
                if cur is not None and pos.shares:
                    cur_val = round(float(pos.shares) * cur, 2)
                    cost_incl_fee = float(pos.entry_total) + float(pos.entry_fee or 0)
                    pnl_abs = round(cur_val - cost_incl_fee, 2)
                open_rows.append(
                    {
                        "id": pos.id,
                        "ticker": pos.ticker,
                        "shares": pos.shares,
                        "entry_price": pos.entry_price,
                        "entry_date": pos.entry_date,
                        "entry_total": pos.entry_total,
                        "entry_fee": pos.entry_fee,
                        "current_price": cur,
                        "current_value": cur_val,
                        "pnl_pct": pnl_pct,
                        "pnl_abs": pnl_abs,
                        "status": pos.status,
                    }
                )
            else:
                closed_rows.append(
                    {
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
                )

        return {
            "user_uuid": user.uuid,
            "cash_balance": round(float(pf.cash_balance), 2),
            "open_positions": open_rows,
            "closed_positions": closed_rows,
        }

    def _prices_map(self, tickers: list[str]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for t in tickers:
            out[t] = _current_price_for_ticker(self._data, t)
        return out

    async def get_summary(
        self,
        session: AsyncSession,
        user: UserProfile,
    ) -> dict[str, Any]:
        overview = await self.get_overview(session, user)
        cash = overview["cash_balance"]
        open_pos = overview["open_positions"]
        closed_pos = overview["closed_positions"]

        unrealized = 0.0
        for p in open_pos:
            if p.get("pnl_abs") is not None:
                unrealized += p["pnl_abs"]

        realized = 0.0
        total_fees = 0.0
        for p in closed_pos:
            if p.get("pnl_abs") is not None:
                realized += p["pnl_abs"]
            total_fees += float(p.get("entry_fee") or 0)
            total_fees += float(p.get("exit_fee") or 0)
        for p in open_pos:
            total_fees += float(p.get("entry_fee") or 0)

        invested_open = sum(float(p.get("entry_total") or 0) for p in open_pos)
        current_open_value = sum(
            float(p["current_value"])
            for p in open_pos
            if p.get("current_value") is not None
        )

        total_value = cash + current_open_value

        return {
            "cash_balance": cash,
            "total_portfolio_value": round(total_value, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(realized, 2),
            "total_fees_paid": round(total_fees, 2),
            "open_notional_at_cost": round(invested_open, 2),
            "open_market_value": round(current_open_value, 2),
            "n_open": len(open_pos),
            "n_closed": len(closed_pos),
        }

    async def deposit(
        self,
        session: AsyncSession,
        user: UserProfile,
        amount: float,
    ) -> UserPortfolio:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        pf.cash_balance = round(float(pf.cash_balance) + amount, 2)
        session.add(
            CashTransaction(
                portfolio_id=pf.id,
                amount=amount,
                tx_type="deposit",
                description="Cash deposit",
            )
        )
        await session.commit()
        await session.refresh(pf)
        return pf

    async def add_position(
        self,
        session: AsyncSession,
        user: UserProfile,
        *,
        ticker: str,
        shares: int,
        entry_price: float,
        entry_date: str,
    ) -> UserPosition:
        if shares < 1:
            raise ValueError("shares must be >= 1")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        t = ticker.strip().upper()
        if not t:
            raise ValueError("ticker required")

        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        entry_total = round(float(shares) * float(entry_price), 2)
        fee = swissquote_fee(entry_total)
        cost = round(entry_total + fee, 2)

        if float(pf.cash_balance) < cost:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f} CHF (incl. fee), have {pf.cash_balance:.2f}"
            )

        pos = UserPosition(
            portfolio_id=pf.id,
            ticker=t,
            shares=int(shares),
            entry_price=float(entry_price),
            entry_date=entry_date,
            entry_total=entry_total,
            entry_fee=fee,
            status="open",
        )
        pf.cash_balance = round(float(pf.cash_balance) - cost, 2)
        session.add(pos)
        session.add(
            CashTransaction(
                portfolio_id=pf.id,
                amount=-cost,
                tx_type="buy",
                description=f"Buy {shares} {t} @ {entry_price}",
            )
        )
        await session.commit()
        await session.refresh(pos)
        return pos

    async def close_position(
        self,
        session: AsyncSession,
        user: UserProfile,
        position_id: int,
        exit_price: float,
        exit_date: str | None,
    ) -> UserPosition:
        if exit_price <= 0:
            raise ValueError("exit_price must be positive")
        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        stmt = select(UserPosition).where(
            UserPosition.id == position_id,
            UserPosition.portfolio_id == pf.id,
        )
        res = await session.execute(stmt)
        pos = res.scalar_one_or_none()
        if pos is None:
            raise ValueError("Position not found")
        if pos.status != "open":
            raise ValueError("Position is not open")

        exit_total = round(float(pos.shares) * float(exit_price), 2)
        exit_fee = swissquote_fee(exit_total)
        proceeds = round(exit_total - exit_fee, 2)

        pos.exit_price = float(exit_price)
        pos.exit_date = exit_date or date.today().isoformat()
        pos.exit_total = exit_total
        pos.exit_fee = exit_fee
        pos.status = "closed"

        cost_basis = float(pos.entry_total) + float(pos.entry_fee)
        pos.pnl_abs = round(proceeds - cost_basis, 2)
        if pos.entry_total and pos.entry_total > 0:
            pos.pnl_pct = round(pos.pnl_abs / pos.entry_total, 6)
        else:
            pos.pnl_pct = None

        pf.cash_balance = round(float(pf.cash_balance) + proceeds, 2)
        session.add(
            CashTransaction(
                portfolio_id=pf.id,
                amount=proceeds,
                tx_type="sell",
                description=f"Sell {pos.shares} {pos.ticker} @ {exit_price}",
            )
        )
        await session.commit()
        await session.refresh(pos)
        return pos

    async def delete_position(
        self,
        session: AsyncSession,
        user: UserProfile,
        position_id: int,
    ) -> None:
        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        stmt = select(UserPosition).where(
            UserPosition.id == position_id,
            UserPosition.portfolio_id == pf.id,
        )
        res = await session.execute(stmt)
        pos = res.scalar_one_or_none()
        if pos is None:
            raise ValueError("Position not found")

        if pos.status == "open":
            # Refund entry notional only; entry fee is not refunded (already paid).
            refund = float(pos.entry_total)
            pf.cash_balance = round(float(pf.cash_balance) + refund, 2)
            session.add(
                CashTransaction(
                    portfolio_id=pf.id,
                    amount=refund,
                    tx_type="delete_open_refund",
                    description=f"Delete open {pos.ticker} — refund entry notional",
                )
            )
        else:
            # Undo round-trip cash effect: remove sell proceeds, restore buy cost.
            et = float(pos.entry_total) + float(pos.entry_fee or 0)
            ex = float(pos.exit_total or 0) - float(pos.exit_fee or 0)
            pf.cash_balance = round(float(pf.cash_balance) - ex + et, 2)
            session.add(
                CashTransaction(
                    portfolio_id=pf.id,
                    amount=-ex + et,
                    tx_type="delete_closed_reversal",
                    description=f"Delete closed {pos.ticker} — reverse cash effect",
                )
            )

        await session.delete(pos)
        await session.commit()
