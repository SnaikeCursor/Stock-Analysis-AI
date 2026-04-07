"""Portfolio Service — CRUD operations on portfolio positions and P&L tracking.

Bridges the signal generation layer (ModelService) with the persistence
layer (SQLAlchemy models) and live market data (DataService) for real-time
P&L computation.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db import (
    Alert,
    AlertType,
    PortfolioPosition,
    RebalanceProposal,
    RebalanceProposalStatus,
    Signal,
    SignalStatus,
)
from backend.services.data_service import DataService

logger = logging.getLogger(__name__)

_ALLOWED_POSITION_PATCH_KEYS = frozenset(
    {"shares", "entry_price", "entry_date", "exit_price", "exit_date"}
)


def _sync_position_derived_fields(pos: PortfolioPosition) -> None:
    """Recompute entry_total, exit_total, and pnl_pct from core position fields."""
    if pos.shares is not None and pos.entry_price is not None:
        pos.entry_total = round(float(pos.shares) * float(pos.entry_price), 2)
    else:
        pos.entry_total = None

    if pos.shares is not None and pos.exit_price is not None:
        pos.exit_total = round(float(pos.shares) * float(pos.exit_price), 2)
    else:
        pos.exit_total = None

    if (
        pos.exit_price is not None
        and pos.entry_price is not None
        and float(pos.entry_price) > 0
    ):
        pos.pnl_pct = round(
            (float(pos.exit_price) - float(pos.entry_price)) / float(pos.entry_price),
            6,
        )
    else:
        pos.pnl_pct = None


class PortfolioService:
    """Async service for portfolio state management backed by SQLite."""

    def __init__(self, data_service: DataService) -> None:
        self._data = data_service

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    async def save_signal(
        self,
        session: AsyncSession,
        *,
        cutoff_date: str,
        regime_label: str,
        regime_confidence: float,
        portfolio_json: str,
        portfolio: list[dict[str, Any]],
        status: SignalStatus = SignalStatus.ACTIVE,
    ) -> Signal:
        """Persist a generated signal and create linked positions.

        * ``ACTIVE`` — expires any previously active signal (scheduler / legacy flow).
        * ``PENDING`` — does not touch active portfolio; expires prior pending drafts only.
        """
        if status == SignalStatus.ACTIVE:
            await self._expire_active_signals(session)
        elif status == SignalStatus.PENDING:
            await self._expire_pending_signals(session)

        signal = Signal(
            cutoff_date=cutoff_date,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            portfolio_json=portfolio_json,
            status=status,
        )
        session.add(signal)
        await session.flush()

        for entry in portfolio:
            position = PortfolioPosition(
                signal_id=signal.id,
                ticker=entry["ticker"],
                weight=entry["weight"],
                entry_date=cutoff_date,
            )
            session.add(position)

        alert = Alert(
            type=AlertType.SIGNAL_GENERATED,
            message=f"New signal generated ({cutoff_date}): "
            f"{len(portfolio)} positions, regime={regime_label}",
        )
        session.add(alert)

        await session.commit()
        logger.info("Signal %d saved with %d positions", signal.id, len(portfolio))
        return signal

    def get_current_prices_for_tickers(self, tickers: list[str]) -> dict[str, float | None]:
        """Latest close from cached OHLCV per ticker; ``None`` if unavailable."""
        if not tickers:
            return {}
        self._data.ensure_data_covers(date.today().isoformat())
        out: dict[str, float | None] = {}
        for t in tickers:
            try:
                df = self._data.get_ticker_price(t)
                if not df.empty and "Close" in df.columns:
                    out[t] = round(float(df["Close"].iloc[-1]), 2)
                else:
                    out[t] = None
            except (KeyError, Exception) as exc:
                logger.debug("Price lookup failed for %s: %s", t, exc)
                out[t] = None
        return out

    async def get_latest_signal(self, session: AsyncSession) -> Signal | None:
        """Return the most recent active signal, or ``None``."""
        stmt = (
            select(Signal)
            .where(Signal.status == SignalStatus.ACTIVE)
            .order_by(Signal.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_signal_history(self, session: AsyncSession) -> list[Signal]:
        """Return all signals ordered by creation date (newest first)."""
        stmt = select(Signal).order_by(Signal.created_at.desc())
        result = await session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def get_active_positions(
        self,
        session: AsyncSession,
    ) -> list[PortfolioPosition]:
        """Return positions belonging to the currently active signal."""
        signal = await self.get_latest_signal(session)
        if signal is None:
            return []
        stmt = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == signal.id)
            .order_by(PortfolioPosition.weight.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def close_position(
        self,
        session: AsyncSession,
        position_id: int,
        exit_price: float,
        exit_date: str,
    ) -> PortfolioPosition:
        """Mark a position as closed with exit price and date."""
        stmt = select(PortfolioPosition).where(PortfolioPosition.id == position_id)
        result = await session.execute(stmt)
        pos = result.scalar_one()

        pos.exit_price = exit_price
        pos.exit_date = exit_date
        _sync_position_derived_fields(pos)

        await session.commit()
        return pos

    async def update_position(
        self,
        session: AsyncSession,
        position_id: int,
        updates: dict[str, Any],
    ) -> PortfolioPosition:
        """Apply partial updates to a position and refresh totals / P&L."""
        if not updates:
            raise ValueError("No fields to update")

        bad = set(updates) - _ALLOWED_POSITION_PATCH_KEYS
        if bad:
            raise ValueError(f"Unsupported fields: {', '.join(sorted(bad))}")

        stmt = select(PortfolioPosition).where(PortfolioPosition.id == position_id)
        result = await session.execute(stmt)
        pos = result.scalar_one_or_none()
        if pos is None:
            raise ValueError(f"No position with id {position_id}")

        for key, value in updates.items():
            setattr(pos, key, value)

        _sync_position_derived_fields(pos)

        await session.commit()
        await session.refresh(pos)
        return pos

    async def compute_live_pnl(
        self,
        session: AsyncSession,
    ) -> list[dict[str, Any]]:
        """Compute unrealised P&L for active positions using latest cached OHLCV.

        Returns a list of dicts with ticker, weight, entry_price,
        current_price, and pnl_pct.
        """
        positions = await self.get_active_positions(session)
        if not positions:
            return []

        self._data.ensure_data_covers(date.today().isoformat())

        results: list[dict[str, Any]] = []
        for pos in positions:
            current_price: float | None = None
            pnl_pct: float | None = None

            try:
                df = self._data.get_ticker_price(pos.ticker)
                if not df.empty and "Close" in df.columns:
                    current_price = round(float(df["Close"].iloc[-1]), 2)
            except (KeyError, Exception) as exc:
                logger.debug("Price lookup failed for %s: %s", pos.ticker, exc)

            if pos.entry_price and current_price and pos.entry_price > 0:
                pnl_pct = round((current_price - pos.entry_price) / pos.entry_price, 6)

            entry_total = pos.entry_total
            if entry_total is None and pos.shares and pos.entry_price:
                entry_total = round(float(pos.shares) * float(pos.entry_price), 2)

            current_value: float | None = None
            if pos.shares and current_price:
                current_value = round(float(pos.shares) * current_price, 2)

            pnl_abs: float | None = None
            if current_value is not None and entry_total is not None:
                pnl_abs = round(current_value - entry_total, 2)

            results.append(
                {
                    "id": pos.id,
                    "ticker": pos.ticker,
                    "weight": pos.weight,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date,
                    "entry_total": entry_total,
                    "current_price": current_price,
                    "current_value": current_value,
                    "pnl_pct": pnl_pct,
                    "pnl_abs": pnl_abs,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    async def get_unread_alerts(self, session: AsyncSession) -> list[Alert]:
        """Return all unread alerts ordered by creation time (newest first)."""
        stmt = (
            select(Alert)
            .where(Alert.read == False)  # noqa: E712
            .order_by(Alert.created_at.desc())
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_alerts(self, session: AsyncSession) -> list[Alert]:
        """Return all alerts ordered by creation time (newest first)."""
        stmt = select(Alert).order_by(Alert.created_at.desc())
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def mark_alert_read(self, session: AsyncSession, alert_id: int) -> Alert:
        """Mark a single alert as read."""
        stmt = select(Alert).where(Alert.id == alert_id)
        result = await session.execute(stmt)
        alert = result.scalar_one()
        alert.read = True
        await session.commit()
        return alert

    async def create_alert(
        self,
        session: AsyncSession,
        *,
        alert_type: AlertType,
        message: str,
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(type=alert_type, message=message)
        session.add(alert)
        await session.commit()
        return alert

    # ------------------------------------------------------------------
    # Rebalancing (quarterly scheduler vs active portfolio)
    # ------------------------------------------------------------------

    def compute_rebalance_instructions(
        self,
        *,
        old_positions: list[PortfolioPosition],
        new_portfolio: list[dict[str, Any]],
        current_prices: dict[str, float | None],
    ) -> dict[str, Any]:
        """Compare current share holdings to target weights; return JSON-ready instructions."""
        weights = {e["ticker"]: float(e["weight"]) for e in new_portfolio}
        new_tickers = set(weights.keys())

        open_positions = [p for p in old_positions if p.exit_price is None]

        current_by_ticker: dict[str, dict[str, Any]] = {}
        total_value = 0.0
        for pos in open_positions:
            sh = pos.shares
            if sh is None or sh <= 0:
                continue
            price = current_prices.get(pos.ticker)
            if price is None or price <= 0:
                continue
            value = float(sh) * float(price)
            total_value += value
            current_by_ticker[pos.ticker] = {
                "shares": int(sh),
                "price": float(price),
                "value": value,
            }

        instructions: list[dict[str, Any]] = []

        if total_value <= 0:
            return {
                "portfolio_value": 0.0,
                "swap_count": 0,
                "instructions": [],
                "note": (
                    "Keine offenen Stueckpositionen mit Kurs — "
                    "Portfolio aktivieren bzw. Stueckzahlen erfassen fuer Swap-Vorschlaege."
                ),
            }

        # Exit names no longer in the new signal
        for pos in open_positions:
            if pos.ticker in new_tickers:
                continue
            sh = pos.shares
            if sh is None or sh <= 0:
                continue
            price = current_prices.get(pos.ticker)
            if price is None or price <= 0:
                continue
            n = int(sh)
            instructions.append(
                {
                    "ticker": pos.ticker,
                    "action": "sell",
                    "shares": n,
                    "estimated_price": round(float(price), 2),
                    "estimated_value": round(n * float(price), 2),
                    "reason": "not_in_new_signal",
                }
            )

        for ticker in sorted(new_tickers):
            w = weights[ticker]
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                instructions.append(
                    {
                        "ticker": ticker,
                        "action": "skip",
                        "shares": 0,
                        "reason": "no_current_price",
                    }
                )
                continue
            target_value = total_value * w
            target_shares = int(math.floor(target_value / float(price)))
            cur_sh = int(current_by_ticker.get(ticker, {}).get("shares", 0))
            delta = target_shares - cur_sh
            est = round(float(price), 2)
            if delta > 0:
                instructions.append(
                    {
                        "ticker": ticker,
                        "action": "buy",
                        "shares": delta,
                        "estimated_price": est,
                        "estimated_value": round(delta * float(price), 2),
                    }
                )
            elif delta < 0:
                instructions.append(
                    {
                        "ticker": ticker,
                        "action": "sell",
                        "shares": abs(delta),
                        "estimated_price": est,
                        "estimated_value": round(abs(delta) * float(price), 2),
                    }
                )
            else:
                instructions.append(
                    {
                        "ticker": ticker,
                        "action": "hold",
                        "shares": cur_sh,
                        "estimated_price": est,
                        "estimated_value": round(cur_sh * float(price), 2),
                    }
                )

        swap_count = sum(
            1
            for i in instructions
            if i.get("action") in ("buy", "sell") and int(i.get("shares") or 0) > 0
        )

        out = {
            "portfolio_value": round(total_value, 2),
            "swap_count": swap_count,
            "instructions": instructions,
        }
        return out

    def _rebalance_alert_summary(self, payload: dict[str, Any], max_items: int = 4) -> str:
        """Short human-readable line for alert body."""
        parts: list[str] = []
        for row in payload.get("instructions", [])[:max_items]:
            act = row.get("action")
            t = row.get("ticker")
            sh = row.get("shares")
            est = row.get("estimated_price")
            if act in ("buy", "sell") and sh and est is not None:
                verb = "Kauf" if act == "buy" else "Verkauf"
                parts.append(f"{verb} {sh}x {t} zu ~{est}")
        if not parts:
            return ""
        return " | ".join(parts)

    async def create_rebalance_proposal_for_new_signal(
        self,
        session: AsyncSession,
        *,
        new_signal_id: int,
    ) -> RebalanceProposal | None:
        """If an ACTIVE portfolio exists, link it to the new PENDING signal with swap JSON + alert."""
        stmt_new = select(Signal).where(Signal.id == new_signal_id)
        res_new = await session.execute(stmt_new)
        new_signal = res_new.scalar_one_or_none()
        if new_signal is None:
            return None

        old_signal = await self.get_latest_signal(session)
        if old_signal is None or old_signal.id == new_signal_id:
            return None

        stmt_pos = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == old_signal.id)
            .order_by(PortfolioPosition.ticker)
        )
        res_pos = await session.execute(stmt_pos)
        old_positions = list(res_pos.scalars().all())

        try:
            new_pf = json.loads(new_signal.portfolio_json)
        except (json.JSONDecodeError, TypeError):
            new_pf = []

        tickers = set()
        for p in old_positions:
            tickers.add(p.ticker)
        for entry in new_pf:
            tickers.add(entry["ticker"])

        prices = self.get_current_prices_for_tickers(sorted(tickers))
        payload = self.compute_rebalance_instructions(
            old_positions=old_positions,
            new_portfolio=new_pf,
            current_prices=prices,
        )

        proposal = RebalanceProposal(
            old_signal_id=old_signal.id,
            new_signal_id=new_signal_id,
            status=RebalanceProposalStatus.PENDING,
            instructions_json=json.dumps(payload, ensure_ascii=False),
        )
        session.add(proposal)

        sc = int(payload.get("swap_count") or 0)
        pv = payload.get("portfolio_value")
        detail = self._rebalance_alert_summary(payload)
        if sc == 0:
            msg = (
                f"Rebalancing pruefen: neues Quartalssignal ({new_signal.cutoff_date}) "
                f"steht bereit — keine automatischen Swaps (Portfoliowert {pv}). "
                f"{payload.get('note') or ''}"
            ).strip()
        else:
            msg = (
                f"Rebalancing faellig: {sc} Swap(s) noetig — geschaetzter Portfoliowert {pv} CHF. "
                f"Neues Signal {new_signal.cutoff_date}. "
            )
            if detail:
                msg += detail

        alert = Alert(type=AlertType.REBALANCING_DUE, message=msg.strip())
        session.add(alert)

        await session.commit()
        await session.refresh(proposal)
        logger.info(
            "Rebalance proposal %d: old=%d new=%d swaps=%d",
            proposal.id,
            old_signal.id,
            new_signal_id,
            sc,
        )
        return proposal

    async def get_pending_rebalance_proposal(
        self,
        session: AsyncSession,
    ) -> RebalanceProposal | None:
        """Return the most recent PENDING rebalance proposal, or ``None``."""
        stmt = (
            select(RebalanceProposal)
            .where(RebalanceProposal.status == RebalanceProposalStatus.PENDING)
            .order_by(RebalanceProposal.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def dismiss_rebalance_proposal(
        self,
        session: AsyncSession,
        proposal_id: int,
    ) -> RebalanceProposal:
        """Mark a proposal as DISMISSED (user declines the rebalance)."""
        stmt = select(RebalanceProposal).where(RebalanceProposal.id == proposal_id)
        result = await session.execute(stmt)
        proposal = result.scalar_one_or_none()
        if proposal is None:
            raise ValueError(f"RebalanceProposal {proposal_id} not found")
        proposal.status = RebalanceProposalStatus.DISMISSED
        await session.commit()
        await session.refresh(proposal)
        return proposal

    async def execute_rebalance(
        self,
        session: AsyncSession,
        *,
        rebalance_id: int,
        executed_trades: list[dict[str, Any]],
    ) -> Signal:
        """Execute a pending rebalance proposal with user-confirmed trades.

        Workflow:
        1. Validate proposal exists and is PENDING.
        2. Close old signal positions using actual sell prices from trades.
        3. Stamp new signal positions with actual buy/hold data from trades.
        4. Transition old signal → EXPIRED, new signal → ACTIVE.
        5. Mark proposal → EXECUTED.

        Args:
            rebalance_id: ID of the PENDING ``RebalanceProposal``.
            executed_trades: User-confirmed trades, each containing
                ``{ticker, action, shares, price, date}``.

        Returns:
            The now-ACTIVE ``Signal`` instance.

        Raises:
            ValueError: If the proposal or signals are in unexpected states.
        """
        stmt = select(RebalanceProposal).where(RebalanceProposal.id == rebalance_id)
        result = await session.execute(stmt)
        proposal = result.scalar_one_or_none()
        if proposal is None:
            raise ValueError(f"RebalanceProposal {rebalance_id} not found")

        prop_status = (
            proposal.status.value
            if hasattr(proposal.status, "value")
            else proposal.status
        )
        if prop_status != RebalanceProposalStatus.PENDING.value:
            raise ValueError(
                f"Proposal {rebalance_id} is '{prop_status}', expected 'pending'"
            )

        old_signal_id = proposal.old_signal_id
        new_signal_id = proposal.new_signal_id

        stmt_old = select(Signal).where(Signal.id == old_signal_id)
        old_signal = (await session.execute(stmt_old)).scalar_one_or_none()
        if old_signal is None:
            raise ValueError(f"Old signal {old_signal_id} not found")

        stmt_new = select(Signal).where(Signal.id == new_signal_id)
        new_signal = (await session.execute(stmt_new)).scalar_one_or_none()
        if new_signal is None:
            raise ValueError(f"New signal {new_signal_id} not found")

        trades_by_ticker: dict[str, dict[str, Any]] = {}
        for t in executed_trades:
            trades_by_ticker[t["ticker"]] = t

        # --- close old positions ---
        stmt_old_pos = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == old_signal_id)
        )
        old_positions = list(
            (await session.execute(stmt_old_pos)).scalars().all()
        )

        for pos in old_positions:
            if pos.exit_price is not None:
                continue
            trade = trades_by_ticker.get(pos.ticker)
            if trade and trade.get("action") == "sell":
                pos.exit_price = float(trade["price"])
                pos.exit_date = trade.get("date") or date.today().isoformat()
            else:
                price = self.get_current_prices_for_tickers([pos.ticker]).get(pos.ticker)
                pos.exit_price = price
                pos.exit_date = date.today().isoformat()
            _sync_position_derived_fields(pos)

        old_signal.status = SignalStatus.EXPIRED

        # --- stamp new positions with trade data ---
        stmt_new_pos = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == new_signal_id)
        )
        new_positions = list(
            (await session.execute(stmt_new_pos)).scalars().all()
        )

        for pos in new_positions:
            trade = trades_by_ticker.get(pos.ticker)
            if trade:
                action = trade.get("action", "buy")
                if action in ("buy", "hold"):
                    pos.shares = int(trade["shares"])
                    pos.entry_price = float(trade["price"])
                    pos.entry_date = trade.get("date") or date.today().isoformat()
                    _sync_position_derived_fields(pos)

        new_signal.status = SignalStatus.ACTIVE
        proposal.status = RebalanceProposalStatus.EXECUTED

        await session.commit()
        await session.refresh(new_signal)
        logger.info(
            "Rebalance %d executed: old=%d→expired, new=%d→active, %d trades",
            rebalance_id,
            old_signal_id,
            new_signal_id,
            len(executed_trades),
        )
        return new_signal

    # ------------------------------------------------------------------
    # Portfolio activation (user confirms purchases)
    # ------------------------------------------------------------------

    async def activate_portfolio(
        self,
        session: AsyncSession,
        *,
        signal_id: int,
        investment_amount: float,
        positions: list[dict[str, Any]],
    ) -> Signal:
        """Activate a PENDING signal: close old positions, stamp new ones with shares.

        Args:
            signal_id: ID of the PENDING signal to activate.
            investment_amount: Total CHF invested (informational; actual amounts
                come from the per-position ``shares * entry_price``).
            positions: List of ``{ticker, shares, entry_price, entry_date}`` dicts
                supplied by the user after optional manual adjustments.

        Returns:
            The now-ACTIVE ``Signal`` instance.

        Raises:
            ValueError: If the signal doesn't exist, isn't PENDING, or tickers mismatch.
        """
        stmt = select(Signal).where(Signal.id == signal_id)
        result = await session.execute(stmt)
        target_signal = result.scalar_one_or_none()
        if target_signal is None:
            raise ValueError(f"Signal {signal_id} not found")
        status_val = (
            target_signal.status.value
            if hasattr(target_signal.status, "value")
            else target_signal.status
        )
        if status_val != SignalStatus.PENDING.value:
            raise ValueError(
                f"Signal {signal_id} is '{status_val}', expected 'pending'"
            )

        # --- close old active portfolio ---
        old_signal = await self.get_latest_signal(session)
        if old_signal is not None:
            await self._close_active_positions_at_market(session, old_signal)
            old_signal.status = SignalStatus.EXPIRED

        # --- stamp the new positions with user-confirmed trade data ---
        pos_stmt = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == signal_id)
        )
        pos_result = await session.execute(pos_stmt)
        existing_positions = {p.ticker: p for p in pos_result.scalars().all()}

        incoming_tickers = {p["ticker"] for p in positions}
        known_tickers = set(existing_positions.keys())
        unknown = incoming_tickers - known_tickers
        if unknown:
            raise ValueError(
                f"Tickers not in signal: {', '.join(sorted(unknown))}"
            )

        for entry in positions:
            pos = existing_positions.get(entry["ticker"])
            if pos is None:
                continue
            pos.shares = int(entry["shares"])
            pos.entry_price = float(entry["entry_price"])
            pos.entry_date = entry.get("entry_date") or date.today().isoformat()
            _sync_position_derived_fields(pos)

        target_signal.status = SignalStatus.ACTIVE

        await session.commit()
        await session.refresh(target_signal)
        logger.info(
            "Signal %d activated — %d positions, investment %.2f",
            signal_id,
            len(positions),
            investment_amount,
        )
        return target_signal

    async def _close_active_positions_at_market(
        self,
        session: AsyncSession,
        signal: Signal,
    ) -> None:
        """Close all open positions of *signal* using current market prices."""
        stmt = (
            select(PortfolioPosition)
            .where(PortfolioPosition.signal_id == signal.id)
        )
        result = await session.execute(stmt)
        positions = list(result.scalars().all())

        tickers = [p.ticker for p in positions if p.exit_price is None]
        if not tickers:
            return

        prices = self.get_current_prices_for_tickers(tickers)
        today = date.today().isoformat()

        for pos in positions:
            if pos.exit_price is not None:
                continue
            price = prices.get(pos.ticker)
            if price is not None:
                pos.exit_price = price
            pos.exit_date = today
            _sync_position_derived_fields(pos)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _expire_active_signals(self, session: AsyncSession) -> int:
        """Set all currently active signals to expired. Returns count updated."""
        stmt = (
            update(Signal)
            .where(Signal.status == SignalStatus.ACTIVE)
            .values(status=SignalStatus.EXPIRED)
        )
        result = await session.execute(stmt)
        return result.rowcount

    async def _expire_pending_signals(self, session: AsyncSession) -> int:
        """Set all pending (draft) signals to expired so only one draft exists."""
        stmt = (
            update(Signal)
            .where(Signal.status == SignalStatus.PENDING)
            .values(status=SignalStatus.EXPIRED)
        )
        result = await session.execute(stmt)
        return result.rowcount
