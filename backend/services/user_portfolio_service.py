"""User-owned manual portfolio: cash, positions, Swissquote-style fees, live P&L."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models.db import (
    CashTransaction,
    Signal,
    UserPortfolio,
    UserPosition,
    UserProfile,
)
from backend.portfolio_json import parse_portfolio_json
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

    async def apply_signal(
        self,
        session: AsyncSession,
        user: UserProfile,
        *,
        signal_id: int,
        investment_amount: float,
    ) -> dict[str, Any]:
        """Buy the positions recommended by a signal into the user portfolio.

        Calculates shares per position based on weight and current prices,
        deducts cash (notional + Swissquote fees), and creates UserPositions.
        """
        stmt = select(Signal).where(Signal.id == signal_id)
        res = await session.execute(stmt)
        signal = res.scalar_one_or_none()
        if signal is None:
            raise ValueError(f"Signal {signal_id} not found")

        portfolio_entries, _ = parse_portfolio_json(signal.portfolio_json)
        if not portfolio_entries:
            raise ValueError("Signal has no positions")

        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        cash = float(pf.cash_balance)
        if investment_amount > cash:
            raise ValueError(
                f"Insufficient cash: want to invest {investment_amount:.2f} CHF "
                f"but only {cash:.2f} available"
            )

        tickers = [e["ticker"] for e in portfolio_entries]
        prices = await asyncio.to_thread(self._prices_map, tickers)

        today = date.today().isoformat()
        created: list[dict[str, Any]] = []
        total_cost = 0.0

        for entry in portfolio_entries:
            ticker = entry["ticker"]
            weight = float(entry["weight"])
            price = prices.get(ticker)
            if price is None or price <= 0:
                logger.warning("Skipping %s — no current price available", ticker)
                continue

            notional = investment_amount * weight
            shares = math.floor(notional / price)
            if shares < 1:
                logger.info("Skipping %s — weight too small for even 1 share", ticker)
                continue

            entry_total = round(shares * price, 2)
            fee = swissquote_fee(entry_total)
            cost = round(entry_total + fee, 2)

            if total_cost + cost > cash:
                logger.warning(
                    "Skipping %s — would exceed cash (need %.2f, remaining %.2f)",
                    ticker, cost, cash - total_cost,
                )
                continue

            pos = UserPosition(
                portfolio_id=pf.id,
                ticker=ticker,
                shares=shares,
                entry_price=price,
                entry_date=today,
                entry_total=entry_total,
                entry_fee=fee,
                status="open",
            )
            session.add(pos)
            session.add(
                CashTransaction(
                    portfolio_id=pf.id,
                    amount=-cost,
                    tx_type="buy",
                    description=f"Signal #{signal_id}: Buy {shares} {ticker} @ {price:.2f}",
                )
            )
            total_cost += cost
            created.append({
                "ticker": ticker,
                "shares": shares,
                "entry_price": price,
                "entry_total": entry_total,
                "fee": fee,
                "weight": weight,
            })

        pf.cash_balance = round(cash - total_cost, 2)
        signal.status = "active"
        await session.commit()

        logger.info(
            "Signal %d applied to user %s: %d positions, %.2f CHF invested, %.2f CHF remaining",
            signal_id, user.uuid[:8], len(created), total_cost, float(pf.cash_balance),
        )

        return {
            "signal_id": signal_id,
            "positions_created": created,
            "total_invested": round(total_cost, 2),
            "cash_remaining": round(float(pf.cash_balance), 2),
        }

    async def compute_performance(
        self,
        session: AsyncSession,
        user: UserProfile,
    ) -> dict[str, Any]:
        """Compute portfolio performance using Modified Dietz method.

        Accounts for deposits/withdrawals so that cash flows don't distort
        the return calculation.
        """
        pf = await _get_or_create_portfolio(session, user, load_positions=True)

        stmt = (
            select(CashTransaction)
            .where(CashTransaction.portfolio_id == pf.id)
            .order_by(CashTransaction.created_at.asc())
        )
        res = await session.execute(stmt)
        transactions = list(res.scalars().all())

        if not transactions:
            return {
                "total_deposited": 0.0,
                "total_withdrawn": 0.0,
                "current_value": round(float(pf.cash_balance), 2),
                "total_return_pct": None,
                "total_pnl_abs": 0.0,
            }

        total_deposited = 0.0
        total_withdrawn = 0.0
        external_flows: list[tuple[datetime, float]] = []

        for tx in transactions:
            if tx.tx_type == "deposit":
                total_deposited += float(tx.amount)
                external_flows.append((tx.created_at, float(tx.amount)))
            elif tx.tx_type == "withdrawal":
                total_withdrawn += abs(float(tx.amount))
                external_flows.append((tx.created_at, float(tx.amount)))

        tickers = [p.ticker for p in pf.positions if p.status == "open"]
        prices: dict[str, float | None] = {}
        if tickers:
            prices = await asyncio.to_thread(self._prices_map, tickers)

        open_market_value = 0.0
        for pos in pf.positions:
            if pos.status == "open":
                cur = prices.get(pos.ticker)
                if cur is not None:
                    open_market_value += float(pos.shares) * cur

        current_value = float(pf.cash_balance) + open_market_value
        net_external = total_deposited - total_withdrawn

        # Modified Dietz: R = (V_end - V_start - sum(CF)) / (V_start + sum(w_i * CF_i))
        # V_start = 0 for a new portfolio, so we use net_external as the denominator base.
        if not external_flows:
            return_pct = None
        else:
            first_date = external_flows[0][0]
            last_date = datetime.now(first_date.tzinfo) if first_date.tzinfo else datetime.now()
            total_days = max((last_date - first_date).days, 1)

            weighted_cf = 0.0
            for dt, amount in external_flows:
                days_remaining = max((last_date - dt).days, 0)
                w = days_remaining / total_days
                weighted_cf += w * amount

            if weighted_cf > 0:
                return_pct = round((current_value - net_external) / weighted_cf, 6)
            elif net_external > 0:
                return_pct = round((current_value - net_external) / net_external, 6)
            else:
                return_pct = None

        return {
            "total_deposited": round(total_deposited, 2),
            "total_withdrawn": round(total_withdrawn, 2),
            "current_value": round(current_value, 2),
            "open_market_value": round(open_market_value, 2),
            "cash_balance": round(float(pf.cash_balance), 2),
            "net_invested": round(net_external, 2),
            "total_pnl_abs": round(current_value - net_external, 2),
            "total_return_pct": return_pct,
        }

    async def withdraw(
        self,
        session: AsyncSession,
        user: UserProfile,
        amount: float,
    ) -> UserPortfolio:
        """Withdraw cash from portfolio."""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        pf = await _get_or_create_portfolio(session, user, load_positions=False)
        if float(pf.cash_balance) < amount:
            raise ValueError(
                f"Insufficient cash: want {amount:.2f} but only {pf.cash_balance:.2f} available"
            )
        pf.cash_balance = round(float(pf.cash_balance) - amount, 2)
        session.add(
            CashTransaction(
                portfolio_id=pf.id,
                amount=-amount,
                tx_type="withdrawal",
                description="Cash withdrawal",
            )
        )
        await session.commit()
        await session.refresh(pf)
        return pf

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
            # Full reversal: refund entry notional + fee (error correction).
            refund = float(pos.entry_total) + float(pos.entry_fee or 0)
            pf.cash_balance = round(float(pf.cash_balance) + refund, 2)
            session.add(
                CashTransaction(
                    portfolio_id=pf.id,
                    amount=refund,
                    tx_type="delete_open_refund",
                    description=f"Delete open {pos.ticker} — full reversal (entry + fee)",
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
