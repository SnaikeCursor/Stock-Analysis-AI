"""yfinance download and Parquet caching."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance's downloader is not thread-safe; concurrent calls can corrupt frames.
_yf_download_lock = threading.Lock()

__all__ = [
    "download_ohlcv",
    "download_spi_universe",
    "load_fundamentals",
    "default_cache_dir",
]


def default_cache_dir() -> Path:
    """Project ``data/`` directory (created if missing).

    Mirrors ``config.DATA_DIR`` / :func:`config.ensure_data_dir` without requiring
    ``config`` to be importable (same path as ``Path(__file__).parent.parent / "data"``).
    """
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def _safe_ticker_filename(ticker: str) -> str:
    return ticker.replace("/", "_")


def _ohlcv_cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "ohlcv" / f"{_safe_ticker_filename(ticker)}.parquet"


def _fundamentals_cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / "fundamentals" / f"{_safe_ticker_filename(ticker)}.json"


def _ensure_cache_layout(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "ohlcv").mkdir(parents=True, exist_ok=True)
    (cache_dir / "fundamentals").mkdir(parents=True, exist_ok=True)


def _flatten_yf_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex (Price, Ticker) for single-ticker downloads."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    if out.columns.nlevels >= 2:
        out.columns = out.columns.droplevel(-1)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
    return out


def _ohlcv_frame_valid(df: pd.DataFrame) -> bool:
    if df.empty or df.columns.duplicated().any():
        return False
    if "Close" not in df.columns:
        return False
    return bool(df["Close"].notna().any())


def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out


def _read_ohlcv_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _normalize_ohlcv_index(_flatten_yf_ohlcv_columns(df))


def _download_ohlcv_single(
    ticker: str,
    start: str,
    end: str,
    retries: int,
) -> pd.DataFrame:
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            with _yf_download_lock:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                )
        except Exception as exc:
            last_err = repr(exc)
            logger.warning("yfinance download failed for %s (attempt %s): %s", ticker, attempt + 1, exc)
            time.sleep(0.4 * (attempt + 1))
            continue

        if raw is None or raw.empty:
            last_err = "empty response"
            time.sleep(0.2 * (attempt + 1))
            continue

        df = _normalize_ohlcv_index(_flatten_yf_ohlcv_columns(raw))
        if not _ohlcv_frame_valid(df):
            last_err = "empty or invalid OHLCV after normalize"
            continue
        return df

    raise RuntimeError(last_err or "download failed")


def _one_ticker_ohlcv_job(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path,
    force_refresh: bool,
    retries: int,
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Returns (ticker, frame or None, error message or None)."""
    path = _ohlcv_cache_path(cache_dir, ticker)
    if not force_refresh and path.exists():
        try:
            df = _read_ohlcv_parquet(path)
            if not df.empty:
                return ticker, df, None
        except Exception as exc:
            logger.warning("Parquet read failed for %s, re-downloading: %s", ticker, exc)

    try:
        df = _download_ohlcv_single(ticker, start, end, retries=retries)
        try:
            df.to_parquet(path, index=True)
        except Exception as exc:
            logger.warning("Could not write Parquet cache for %s: %s", ticker, exc)
        return ticker, df, None
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning("OHLCV download failed for %s: %s", ticker, msg)
        return ticker, None, msg


def download_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path,
    *,
    force_refresh: bool = False,
    max_workers: int = 8,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV and persist to Parquet cache.

    Each ticker is stored under ``cache_dir/ohlcv/<TICKER>.parquet``. Cached files
    are reused on subsequent runs unless ``force_refresh`` is True.

    Failures (delisted symbols, rate limits, empty history) are logged; only
    successfully loaded tickers appear in the returned mapping.

    Parameters
    ----------
    tickers
        Yahoo Finance symbols (e.g. ``NESN.SW``).
    start, end
        Passed to ``yfinance.download`` (see yfinance docs for exclusivity).
    cache_dir
        Root cache folder (typically :func:`default_cache_dir`).
    force_refresh
        If True, ignore existing Parquet files and re-fetch.
    max_workers
        Thread pool size for scheduling ticker jobs. Actual ``yfinance`` HTTP
        calls are serialized with a lock (the library is not thread-safe).
    retries
        Attempts per ticker after transient failures.
    """
    _ensure_cache_layout(cache_dir)
    if not tickers:
        return {}

    unique = list(dict.fromkeys(tickers))
    out: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                _one_ticker_ohlcv_job,
                t,
                start,
                end,
                cache_dir,
                force_refresh,
                retries,
            ): t
            for t in unique
        }
        for fut in as_completed(futs):
            ticker, df, err = fut.result()
            if df is not None and not df.empty:
                out[ticker] = df
            elif err:
                logger.debug("Skipped %s: %s", ticker, err)

    return out


def download_spi_universe(
    *,
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    max_workers: int = 8,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for the full SPI universe with sensible defaults.

    Uses :func:`src.universe.get_spi_tickers` and falls back to
    ``config.YF_START`` / ``config.YF_END`` when *start* / *end* are omitted.

    Parameters
    ----------
    start, end
        Override the download window (defaults to ``config.YF_START/YF_END``).
    cache_dir
        Root cache folder (defaults to :func:`default_cache_dir`).
    force_refresh
        If True, ignore existing Parquet files and re-fetch.
    max_workers, retries
        Forwarded to :func:`download_ohlcv`.
    """
    from src.universe import get_spi_tickers  # noqa: deferred to avoid circular import

    _start = start
    _end = end
    if _start is None or _end is None:
        try:
            import config as cfg
        except ImportError:
            raise ValueError(
                "start/end are required when the config module is not importable"
            ) from None
        if _start is None:
            _start = cfg.YF_START
        if _end is None:
            _end = cfg.YF_END

    root = cache_dir if cache_dir is not None else default_cache_dir()
    # SMI index for macro regime features (see features.MACRO_BENCHMARK_TICKER); not a stock.
    tickers = list(dict.fromkeys([*get_spi_tickers(), "^SSMI"]))
    logger.info("Downloading SPI universe (%d tickers, %s → %s)", len(tickers), _start, _end)
    return download_ohlcv(
        tickers, _start, _end, root,
        force_refresh=force_refresh, max_workers=max_workers, retries=retries,
    )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def _write_fundamentals_json(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(info)
    path.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")


def _read_fundamentals_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("fundamentals cache is not a JSON object")
    return data


def load_fundamentals(
    ticker: str,
    *,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    retries: int = 2,
) -> dict[str, Any]:
    """Load ``yfinance.Ticker(ticker).info`` fields (P/E, sector, etc.) with JSON cache.

    Cached files live under ``<cache_dir>/fundamentals/<TICKER>.json``. When
    ``cache_dir`` is omitted, :func:`default_cache_dir` is used.

    On repeated failures or empty ``info``, returns ``{}`` and logs a warning.
    """
    root = cache_dir if cache_dir is not None else default_cache_dir()
    _ensure_cache_layout(root)
    path = _fundamentals_cache_path(root, ticker)

    if not force_refresh and path.exists():
        try:
            return _read_fundamentals_json(path)
        except Exception as exc:
            logger.warning("Fundamentals cache read failed for %s: %s", ticker, exc)

    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            with _yf_download_lock:
                t = yf.Ticker(ticker)
                info = t.info
            if not isinstance(info, dict):
                last_err = "info is not a dict"
                time.sleep(0.3 * (attempt + 1))
                continue
            if len(info) == 0:
                last_err = "empty info dict"
                time.sleep(0.3 * (attempt + 1))
                continue
            try:
                _write_fundamentals_json(path, info)
            except Exception as exc:
                logger.warning("Could not write fundamentals cache for %s: %s", ticker, exc)
            return info
        except Exception as exc:
            last_err = repr(exc)
            logger.warning("Fundamentals fetch failed for %s (attempt %s): %s", ticker, attempt + 1, exc)
            time.sleep(0.4 * (attempt + 1))

    logger.warning("Giving up on fundamentals for %s: %s", ticker, last_err)
    return {}
