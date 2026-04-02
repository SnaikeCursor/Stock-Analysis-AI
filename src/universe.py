"""SPI constituent list and liquidity helpers (Yahoo `.SW` tickers).

The full **SPI** universe combines:

* **SMI** (~20 blue-chip names) — :data:`SMI_TICKERS`
* **SPI Extra** (SMI Mid + smaller caps) — :data:`SPI_EXTRA_TICKERS`

The merged result is :data:`SPI_TICKERS` (~215 names before liquidity filter).

Ticker sets are **curated snapshots** derived from the SPI components table on
Wikipedia (``Swiss_Performance_Index``, revision containing the Sept 2020 SPI list).
SMI constituents reflect the ~2020-2024 composition.  Index membership changes over
time; refresh against the SIX Index Data Center for production use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

SPI_EXTRA_TICKERS: list[str] = [
    "ACHI.SW",
    "ADEN.SW",
    "ADVN.SW",
    "ADXN.SW",
    "AEVS.SW",
    "AFP.SW",
    "AIRE.SW",
    "ALLN.SW",
    "ALSN.SW",
    "AMS.SW",
    "APGN.SW",
    "ARBN.SW",
    "ARON.SW",
    "ARYN.SW",
    "ASCN.SW",
    "ASWN.SW",
    "AUTN.SW",
    "BAER.SW",
    "BALN.SW",
    "BANB.SW",
    "BARN.SW",
    "BBN.SW",
    "BCGE.SW",
    "BCHN.SW",
    "BCJ.SW",
    "BCVN.SW",
    "BEAN.SW",
    "BEKN.SW",
    "BELL.SW",
    "BION.SW",
    "BKW.SW",
    "BLKB.SW",
    "BLS.SW",
    "BOBNN.SW",
    "BOSN.SW",
    "BPDG.SW",
    "BRKN.SW",
    "BSKP.SW",
    "BSLN.SW",
    "BUCN.SW",
    "BVZN.SW",
    "CALN.SW",
    "CFT.SW",
    "CICN.SW",
    "CIE.SW",
    "CLN.SW",
    "CLTN.SW",
    "CLXN.SW",
    "CMBN.SW",
    "CON.SW",
    "COPN.SW",
    "COTN.SW",
    "CPHN.SW",
    "DAE.SW",
    "DKSH.SW",
    "DOKA.SW",
    "DUFN.SW",
    "EFGN.SW",
    "ELMN.SW",
    "EMMN.SW",
    "EMSN.SW",
    "ESUN.SW",
    "EVE.SW",
    "FHZN.SW",
    "FI-N.SW",
    "FORN.SW",
    "FREN.SW",
    "FTON.SW",
    "GALE.SW",
    "GAM.SW",
    "GAV.SW",
    "GLKBN.SW",
    "GMI.SW",
    "GRKP.SW",
    "GUR.SW",
    "HBLN.SW",
    "HELN.SW",
    "HIAG.SW",
    "HLEE.SW",
    "HOCN.SW",
    "HREN.SW",
    "HUBN.SW",
    "IDIA.SW",
    "IFCN.SW",
    "IMPN.SW",
    "INA.SW",
    "INRN.SW",
    "IREN.SW",
    "ISN.SW",
    "JFN.SW",
    "KARN.SW",
    "KLIN.SW",
    "KNIN.SW",
    "KOMN.SW",
    "KUD.SW",
    "KURN.SW",
    "LAND.SW",
    "LECN.SW",
    "LEHN.SW",
    "LEON.SW",
    "LINN.SW",
    "LISN.SW",
    "LISP.SW",
    "LLBN.SW",
    "LLQ.SW",
    "LMN.SW",
    "LOGN.SW",
    "LUKN.SW",
    "MBTN.SW",
    "MCHN.SW",
    "MED.SW",
    "METN.SW",
    "MIKN.SW",
    "MOBN.SW",
    "MOLN.SW",
    "MOVE.SW",
    "MOZN.SW",
    "MTG.SW",
    "NREN.SW",
    "NWRN.SW",
    "OBSN.SW",
    "ODHN.SW",
    "OERL.SW",
    "OFN.SW",
    "ORON.SW",
    "PEAN.SW",
    "PEDU.SW",
    "PEHN.SW",
    "PLAN.SW",
    "PM.SW",
    "PMAG.SW",
    "PNHO.SW",
    "POLN.SW",
    "PRFN.SW",
    "PSPN.SW",
    "RIEN.SW",
    "RLF.SW",
    "ROL.SW",
    "ROSE.SW",
    "SAHN.SW",
    "SANN.SW",
    "SCHN.SW",
    "SCHP.SW",
    "SENS.SW",
    "SFPN.SW",
    "SFSN.SW",
    "SFZN.SW",
    "SGKN.SW",
    "SIGN.SW",
    "SKIN.SW",
    "SNBN.SW",
    "SOON.SW",
    "SPCE.SW",
    "SPSN.SW",
    "SQN.SW",
    "SRAIL.SW",
    "STGN.SW",
    "STLN.SW",
    "STMN.SW",
    "STRN.SW",
    "SUN.SW",
    "SWON.SW",
    "SWTQ.SW",
    "TECN.SW",
    "TEMN.SW",
    "TIBN.SW",
    "TKBP.SW",
    "TOHN.SW",
    "TXGN.SW",
    "UBXN.SW",
    "UHRN.SW",
    "VACN.SW",
    "VAHN.SW",
    "VALN.SW",
    "VARN.SW",
    "VATN.SW",
    "VBSN.SW",
    "VET.SW",
    "VIFN.SW",
    "VILN.SW",
    "VLRT.SW",
    "VONN.SW",
    "VPBN.SW",
    "VZN.SW",
    "VZUG.SW",
    "WARN.SW",
    "WIHN.SW",
    "WKB.SW",
    "YPSN.SW",
    "ZEHN.SW",
    "ZG.SW",
    "ZUBN.SW",
    "ZUGN.SW",
    "ZWN.SW",
]

SPI_EXTRA_TICKERS_SET: frozenset[str] = frozenset(SPI_EXTRA_TICKERS)

# ---------------------------------------------------------------------------
# SMI blue-chip constituents (~2020-2024 composition)
# ---------------------------------------------------------------------------

SMI_TICKERS: list[str] = [
    "ABBN.SW",
    "ALC.SW",
    "CFR.SW",
    "CSGN.SW",
    "GEBN.SW",
    "GIVN.SW",
    "HOLN.SW",
    "LONN.SW",
    "NESN.SW",
    "NOVN.SW",
    "PGHN.SW",
    "ROG.SW",
    "SCMN.SW",
    "SGSN.SW",
    "SIKA.SW",
    "SLHN.SW",
    "SREN.SW",
    "UBSG.SW",
    "UHR.SW",
    "ZURN.SW",
]

SMI_TICKERS_SET: frozenset[str] = frozenset(SMI_TICKERS)

# ---------------------------------------------------------------------------
# Full SPI = SMI ∪ SPI Extra (sorted, deduplicated)
# ---------------------------------------------------------------------------

SPI_TICKERS: list[str] = sorted(set(SPI_EXTRA_TICKERS) | set(SMI_TICKERS))
SPI_TICKERS_SET: frozenset[str] = frozenset(SPI_TICKERS)


@dataclass
class SpiCoverageReport:
    """SPI universe vs downloaded OHLCV: gaps and survivorship discussion."""

    universe_size: int
    tickers_with_valid_ohlcv: list[str]
    tickers_missing_file: list[str]
    """In the SPI list but absent from *ohlcv_by_ticker*."""
    tickers_incomplete_ohlcv: list[str]
    """Present but empty, no ``Close``, or fewer than *min_bars* rows."""
    coverage_fraction: float
    bias_interpretation: str
    """Plain-language note on likely direction of survivorship / selection bias."""


SpiExtraCoverageReport = SpiCoverageReport


def report_spi_coverage(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    *,
    min_bars: int = 21,
    tickers_universe: list[str] | None = None,
) -> SpiCoverageReport:
    """Compare the SPI ticker list to cached OHLCV and log gaps.

    **Survivorship / selection bias:** Names with no usable history are often
    delisted, merged, or never listed on Yahoo under the expected symbol.
    Excluding them removes failed names from the backtest universe, which
    typically **upward-biases** realised returns versus a full index that
    included delistings.  Conversely, missing data can also reflect thin
    liquidity or API gaps (not necessarily failure), which weakens the
    direction of bias.

    Parameters
    ----------
    ohlcv_by_ticker
        Ticker → OHLCV as loaded from the project cache.
    min_bars
        Minimum row count to treat OHLCV as *complete* (default 21 trading days).
    tickers_universe
        Override list (defaults to :data:`SPI_TICKERS`).

    Returns
    -------
    SpiCoverageReport
        Sorted ticker lists and a short bias note; also logs missing/incomplete
        tickers at INFO level.
    """
    universe = list(tickers_universe) if tickers_universe is not None else list(
        SPI_TICKERS,
    )
    n_u = len(universe)

    missing: list[str] = []
    incomplete: list[str] = []
    valid: list[str] = []

    for t in universe:
        df = ohlcv_by_ticker.get(t)
        if df is None:
            missing.append(t)
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            incomplete.append(t)
            continue
        if "Close" not in df.columns:
            incomplete.append(t)
            continue
        n_ok = int(df["Close"].dropna().shape[0])
        if n_ok < min_bars:
            incomplete.append(t)
            continue
        valid.append(t)

    missing.sort()
    incomplete.sort()
    valid.sort()

    cov = len(valid) / n_u if n_u else 0.0

    bias = (
        "Universe excludes tickers without usable OHLCV; many gaps are likely "
        "delistings/mergers or bad symbols — that usually tilts backtests toward "
        "survivors and **overstates** average returns vs a full historical index. "
        "Some gaps are data/API noise instead, which is directionally ambiguous."
    )

    report = SpiCoverageReport(
        universe_size=n_u,
        tickers_with_valid_ohlcv=valid,
        tickers_missing_file=missing,
        tickers_incomplete_ohlcv=incomplete,
        coverage_fraction=cov,
        bias_interpretation=bias,
    )

    logger.info(
        "SPI coverage: %d/%d tickers with valid OHLCV (%.1f%%)",
        len(valid),
        n_u,
        100.0 * cov,
    )
    if missing:
        logger.info(
            "SPI — missing from ohlcv map (%d): %s",
            len(missing),
            ", ".join(missing[:40]) + (" …" if len(missing) > 40 else ""),
        )
    if incomplete:
        logger.info(
            "SPI — incomplete OHLCV (%d): %s",
            len(incomplete),
            ", ".join(incomplete[:40]) + (" …" if len(incomplete) > 40 else ""),
        )

    return report


report_spi_extra_coverage = report_spi_coverage


def get_spi_tickers() -> list[str]:
    """Return a copy of the full SPI Yahoo ticker list (sorted, deduplicated)."""
    return list(SPI_TICKERS)


def get_spi_extra_tickers() -> list[str]:
    """Return a copy of the SPI Extra Yahoo ticker list (sorted).

    .. deprecated:: Use :func:`get_spi_tickers` for the full SPI universe.
    """
    return list(SPI_EXTRA_TICKERS)


def get_smi_tickers() -> list[str]:
    """Return a copy of the SMI blue-chip Yahoo ticker list (sorted)."""
    return list(SMI_TICKERS)


def normalize_yahoo_ticker(ticker: str) -> str:
    """Normalize a SIX-style or bare symbol to Yahoo Finance ``*.SW`` form."""
    t = ticker.strip().upper()
    if not t.endswith(".SW"):
        t = f"{t}.SW"
    return t


def is_spi_ticker(ticker: str) -> bool:
    """Return True if ``ticker`` (after normalization) is in the full SPI universe."""
    return normalize_yahoo_ticker(ticker) in SPI_TICKERS_SET


def is_spi_extra_ticker(ticker: str) -> bool:
    """Return True if ``ticker`` (after normalization) is in the SPI Extra universe."""
    return normalize_yahoo_ticker(ticker) in SPI_EXTRA_TICKERS_SET


def is_smi_ticker(ticker: str) -> bool:
    """Return True if ``ticker`` (after normalization) is in the SMI universe."""
    return normalize_yahoo_ticker(ticker) in SMI_TICKERS_SET


def filter_by_min_volume(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    min_daily_volume_chf: float,
    *,
    price_col: str = "Close",
    volume_col: str = "Volume",
) -> list[str]:
    """Return tickers whose mean daily turnover (price × volume) is at least ``min_daily_volume_chf``.

    Swiss listings on SIX are typically CHF-denominated; ``Close`` is treated as the
    CHF price per share. Uses rows where both columns are valid.

    Parameters
    ----------
    ohlcv_by_ticker
        Map ticker -> OHLCV DataFrame (e.g. from ``yfinance`` / project cache).
    min_daily_volume_chf
        Minimum average daily turnover in CHF (e.g. from ``config.MIN_DAILY_VOLUME_CHF``).
    price_col, volume_col
        Column names for notional computation (default ``Close`` × ``Volume``).

    Returns
    -------
    Sorted list of tickers that pass the liquidity filter. Tickers with missing
    columns or empty series are skipped.
    """
    out: list[str] = []
    for ticker, df in ohlcv_by_ticker.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if price_col not in df.columns or volume_col not in df.columns:
            continue
        price = pd.to_numeric(df[price_col], errors="coerce")
        vol = pd.to_numeric(df[volume_col], errors="coerce")
        turnover = (price * vol).dropna()
        if turnover.empty:
            continue
        if float(turnover.mean()) >= min_daily_volume_chf:
            out.append(ticker)
    return sorted(out)
