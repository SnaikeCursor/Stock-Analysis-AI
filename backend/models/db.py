"""SQLAlchemy ORM models and async engine bootstrap for the webapp.

Tables:
  - signals              — generated trading signals with regime context
  - portfolio_positions  — per-ticker entries linked to a signal (shares, P&L)
  - rebalance_proposals  — swap instructions when moving from old to new signal
  - alerts               — rebalancing reminders, regime changes, new signals
  - user_profiles        — anonymous UUID identities (X-User-ID)
  - user_portfolios      — cash balance per user profile
  - user_positions       — manual portfolio positions (open/closed)
  - cash_transactions    — audit log for cash movements
"""

from __future__ import annotations

import enum
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
    sessionmaker,
)

# Relative path resolves against process cwd (Docker WORKDIR /app → /app/data/…).
_DEFAULT_DB = "sqlite+aiosqlite:///data/stock_analysis.db"
DATABASE_URL = (os.environ.get("DATABASE_URL") or _DEFAULT_DB).strip() or _DEFAULT_DB

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class SignalStatus(str, enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"


class RebalanceProposalStatus(str, enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    DISMISSED = "dismissed"


class AlertType(str, enum.Enum):
    REBALANCING_DUE = "rebalancing_due"
    REGIME_CHANGE = "regime_change"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_COVERAGE_SHORTFALL = "signal_coverage_shortfall"


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class Signal(Base):
    __tablename__ = "signals"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    cutoff_date: str = Column(String(10), nullable=False)
    regime_label: str = Column(String(16), nullable=False)
    regime_confidence: float = Column(Float, nullable=False)
    portfolio_json: str = Column(Text, nullable=False)
    status: str = Column(
        Enum(SignalStatus),
        nullable=False,
        default=SignalStatus.PENDING,
    )

    positions = relationship(
        "PortfolioPosition",
        back_populates="signal",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    rebalance_proposals_as_old = relationship(
        "RebalanceProposal",
        foreign_keys="RebalanceProposal.old_signal_id",
        back_populates="old_signal",
    )
    rebalance_proposals_as_new = relationship(
        "RebalanceProposal",
        foreign_keys="RebalanceProposal.new_signal_id",
        back_populates="new_signal",
    )

    def __repr__(self) -> str:
        return (
            f"<Signal id={self.id} cutoff={self.cutoff_date} "
            f"regime={self.regime_label} status={self.status}>"
        )


class PortfolioPosition(Base):
    __tablename__ = "portfolio_positions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    signal_id: int = Column(Integer, ForeignKey("signals.id"), nullable=False, index=True)
    ticker: str = Column(String(20), nullable=False)
    weight: float = Column(Float, nullable=False)
    shares: int | None = Column(Integer, nullable=True)
    entry_price: float | None = Column(Float, nullable=True)
    entry_date: str | None = Column(String(10), nullable=True)
    entry_total: float | None = Column(Float, nullable=True)
    exit_price: float | None = Column(Float, nullable=True)
    exit_date: str | None = Column(String(10), nullable=True)
    exit_total: float | None = Column(Float, nullable=True)
    pnl_pct: float | None = Column(Float, nullable=True)

    signal = relationship("Signal", back_populates="positions")

    def __repr__(self) -> str:
        return (
            f"<Position id={self.id} ticker={self.ticker} "
            f"weight={self.weight:.2%} pnl={self.pnl_pct}>"
        )


class Alert(Base):
    __tablename__ = "alerts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    type: str = Column(Enum(AlertType), nullable=False)
    message: str = Column(Text, nullable=False)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    read: bool = Column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return f"<Alert id={self.id} type={self.type} read={self.read}>"


class RebalanceProposal(Base):
    __tablename__ = "rebalance_proposals"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    old_signal_id: int = Column(Integer, ForeignKey("signals.id"), nullable=False, index=True)
    new_signal_id: int = Column(Integer, ForeignKey("signals.id"), nullable=False, index=True)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    status: str = Column(
        Enum(RebalanceProposalStatus),
        nullable=False,
        default=RebalanceProposalStatus.PENDING,
    )
    instructions_json: str = Column(Text, nullable=False)

    old_signal = relationship(
        "Signal",
        foreign_keys=[old_signal_id],
        back_populates="rebalance_proposals_as_old",
    )
    new_signal = relationship(
        "Signal",
        foreign_keys=[new_signal_id],
        back_populates="rebalance_proposals_as_new",
    )

    def __repr__(self) -> str:
        return (
            f"<RebalanceProposal id={self.id} old={self.old_signal_id} "
            f"new={self.new_signal_id} status={self.status}>"
        )


class UserProfile(Base):
    """Anonymous user identified by a client-generated UUID (header X-User-ID)."""

    __tablename__ = "user_profiles"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    uuid: str = Column(String(36), unique=True, nullable=False, index=True)
    display_name: str | None = Column(String(100), nullable=True)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    portfolio = relationship(
        "UserPortfolio",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )


class UserPortfolio(Base):
    """One cash book per user profile."""

    __tablename__ = "user_portfolios"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: int = Column(Integer, ForeignKey("user_profiles.id"), unique=True, nullable=False)
    cash_balance: float = Column(Float, nullable=False, default=0.0)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    user = relationship("UserProfile", back_populates="portfolio")
    positions = relationship(
        "UserPosition",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    cash_transactions = relationship(
        "CashTransaction",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class UserPosition(Base):
    """Manual long position tracked in CHF with Swissquote-style fees."""

    __tablename__ = "user_positions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: int = Column(Integer, ForeignKey("user_portfolios.id"), nullable=False, index=True)
    ticker: str = Column(String(20), nullable=False)
    shares: int = Column(Integer, nullable=False)
    entry_price: float = Column(Float, nullable=False)
    entry_date: str = Column(String(10), nullable=False)
    entry_total: float = Column(Float, nullable=False)
    entry_fee: float = Column(Float, nullable=False, default=0.0)
    exit_price: float | None = Column(Float, nullable=True)
    exit_date: str | None = Column(String(10), nullable=True)
    exit_total: float | None = Column(Float, nullable=True)
    exit_fee: float | None = Column(Float, nullable=True)
    pnl_abs: float | None = Column(Float, nullable=True)
    pnl_pct: float | None = Column(Float, nullable=True)
    status: str = Column(String(10), nullable=False, default="open")

    portfolio = relationship("UserPortfolio", back_populates="positions")


class CashTransaction(Base):
    """Audit trail for portfolio cash movements."""

    __tablename__ = "cash_transactions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: int = Column(Integer, ForeignKey("user_portfolios.id"), nullable=False, index=True)
    amount: float = Column(Float, nullable=False)
    tx_type: str = Column(String(32), nullable=False)
    description: str | None = Column(Text, nullable=True)
    created_at: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    portfolio = relationship("UserPortfolio", back_populates="cash_transactions")


# ------------------------------------------------------------------
# Database lifecycle helpers
# ------------------------------------------------------------------


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record):
    """Enable WAL mode and foreign keys for every SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _migrate_sqlite_schema(sync_conn) -> None:
    """Add columns introduced after first deploy (SQLite does not migrate via create_all)."""
    if sync_conn.dialect.name != "sqlite":
        return
    rows = sync_conn.execute(text("PRAGMA table_info(portfolio_positions)")).fetchall()
    existing = {row[1] for row in rows}
    for col, sql_type in (
        ("shares", "INTEGER"),
        ("entry_total", "FLOAT"),
        ("exit_total", "FLOAT"),
    ):
        if col not in existing:
            sync_conn.execute(
                text(f"ALTER TABLE portfolio_positions ADD COLUMN {col} {sql_type}")
            )


async def init_db() -> None:
    """Create all tables if they don't exist (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_sqlite_schema)


async def get_session() -> AsyncSession:
    """Yield a new async session (for FastAPI dependency injection)."""
    async with async_session_factory() as session:
        yield session  # type: ignore[misc]
