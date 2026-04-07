"""SQLAlchemy ORM models and async engine bootstrap for the webapp.

Tables:
  - signals              — generated trading signals with regime context
  - portfolio_positions  — per-ticker entries linked to a signal (shares, P&L)
  - rebalance_proposals  — swap instructions when moving from old to new signal
  - alerts               — rebalancing reminders, regime changes, new signals
"""

from __future__ import annotations

import enum
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

DATABASE_URL = "sqlite+aiosqlite:///data/stock_analysis.db"

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
