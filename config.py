"""
Central configuration for the Swiss SPI stock analysis pipeline.

Lookahead: features use only data strictly before the classification window.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- Paths (project root = parent of this file) ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SRC_DIR: Path = PROJECT_ROOT / "src"

# --- yfinance download window ---
# 2010-01-01 gives ~2 years pre-roll (2010-2011) before Q1-2012, the first
# classification period — sufficient for SMA(200) and other long-window indicators.
YF_START: str = "2010-01-01"
YF_END: str = "2025-12-31"

# --- Classification period (Q1 2024): performance label (legacy single-period) ---
CLASS_Q_START: str = "2024-01-02"
CLASS_Q_END: str = "2024-03-31"

# --- Feature cutoff: technicals/fundamentals use data *before* Q1 2024 ---
FEATURE_CUTOFF_DATE: str = "2023-12-31"

# --- Multi-period classification windows for stacked training ---
# Each tuple: (feature_cutoff, q_start, q_end, period_label)
# feature_cutoff is always the last trading day *before* the label window.
# 52 quarters (Q1-2012 → Q4-2024) to maximise regime diversity
# (~15 Bear, ~20 Sideways) and stabilise feature weights.
CLASSIFICATION_PERIODS: list[tuple[str, str, str, str]] = [
    # ── 2012 ──
    ("2011-12-30", "2012-01-02", "2012-03-30", "Q1-2012"),
    ("2012-03-30", "2012-04-02", "2012-06-29", "Q2-2012"),
    ("2012-06-29", "2012-07-02", "2012-09-28", "Q3-2012"),
    ("2012-09-28", "2012-10-01", "2012-12-28", "Q4-2012"),
    # ── 2013 ──
    ("2012-12-28", "2013-01-02", "2013-03-28", "Q1-2013"),
    ("2013-03-28", "2013-04-02", "2013-06-28", "Q2-2013"),
    ("2013-06-28", "2013-07-01", "2013-09-30", "Q3-2013"),
    ("2013-09-30", "2013-10-01", "2013-12-31", "Q4-2013"),
    # ── 2014 ──
    ("2013-12-31", "2014-01-02", "2014-03-31", "Q1-2014"),
    ("2014-03-31", "2014-04-01", "2014-06-30", "Q2-2014"),
    ("2014-06-30", "2014-07-01", "2014-09-30", "Q3-2014"),
    ("2014-09-30", "2014-10-01", "2014-12-31", "Q4-2014"),
    # ── 2015 ──
    ("2014-12-31", "2015-01-02", "2015-03-31", "Q1-2015"),
    ("2015-03-31", "2015-04-01", "2015-06-30", "Q2-2015"),
    ("2015-06-30", "2015-07-01", "2015-09-30", "Q3-2015"),
    ("2015-09-30", "2015-10-01", "2015-12-31", "Q4-2015"),
    # ── 2016 ──
    ("2015-12-31", "2016-01-04", "2016-03-31", "Q1-2016"),
    ("2016-03-31", "2016-04-01", "2016-06-30", "Q2-2016"),
    ("2016-06-30", "2016-07-01", "2016-09-30", "Q3-2016"),
    ("2016-09-30", "2016-10-03", "2016-12-30", "Q4-2016"),
    # ── 2017 ──
    ("2016-12-30", "2017-01-02", "2017-03-31", "Q1-2017"),
    ("2017-03-31", "2017-04-03", "2017-06-30", "Q2-2017"),
    ("2017-06-30", "2017-07-03", "2017-09-29", "Q3-2017"),
    ("2017-09-29", "2017-10-02", "2017-12-29", "Q4-2017"),
    # ── 2018 ──
    ("2017-12-29", "2018-01-02", "2018-03-29", "Q1-2018"),
    ("2018-03-29", "2018-04-03", "2018-06-29", "Q2-2018"),
    ("2018-06-29", "2018-07-02", "2018-09-28", "Q3-2018"),
    ("2018-09-28", "2018-10-01", "2018-12-28", "Q4-2018"),
    # ── 2019 ──
    ("2018-12-28", "2019-01-02", "2019-03-29", "Q1-2019"),
    ("2019-03-29", "2019-04-01", "2019-06-28", "Q2-2019"),
    ("2019-06-28", "2019-07-01", "2019-09-30", "Q3-2019"),
    ("2019-09-30", "2019-10-01", "2019-12-31", "Q4-2019"),
    # ── 2020 ──
    ("2019-12-31", "2020-01-02", "2020-03-31", "Q1-2020"),
    ("2020-03-31", "2020-04-01", "2020-06-30", "Q2-2020"),
    ("2020-06-30", "2020-07-01", "2020-09-30", "Q3-2020"),
    ("2020-09-30", "2020-10-01", "2020-12-31", "Q4-2020"),
    # ── 2021 ──
    ("2020-12-31", "2021-01-04", "2021-03-31", "Q1-2021"),
    ("2021-03-31", "2021-04-01", "2021-06-30", "Q2-2021"),
    ("2021-06-30", "2021-07-01", "2021-09-30", "Q3-2021"),
    ("2021-09-30", "2021-10-01", "2021-12-31", "Q4-2021"),
    # ── 2022 ──
    ("2021-12-31", "2022-01-03", "2022-03-31", "Q1-2022"),
    ("2022-03-31", "2022-04-01", "2022-06-30", "Q2-2022"),
    ("2022-06-30", "2022-07-01", "2022-09-30", "Q3-2022"),
    ("2022-09-30", "2022-10-03", "2022-12-30", "Q4-2022"),
    # ── 2023 ──
    ("2022-12-30", "2023-01-03", "2023-03-31", "Q1-2023"),
    ("2023-03-31", "2023-04-03", "2023-06-30", "Q2-2023"),
    ("2023-06-30", "2023-07-03", "2023-09-29", "Q3-2023"),
    ("2023-09-29", "2023-10-02", "2023-12-29", "Q4-2023"),
    # ── 2024 ──
    ("2023-12-29", "2024-01-02", "2024-03-28", "Q1-2024"),
    ("2024-03-28", "2024-04-02", "2024-06-28", "Q2-2024"),
    ("2024-06-28", "2024-07-01", "2024-09-30", "Q3-2024"),
    ("2024-09-30", "2024-10-01", "2024-12-31", "Q4-2024"),
]

# --- Out-of-sample forward test ---
OOS_YEAR: int = 2025
OOS_FEATURE_CUTOFF_DATE: str = "2024-12-31"  # e.g. Q4 2024 features for 2025 test

# --- Universe filtering ---
MIN_DAILY_VOLUME_CHF: float = 50_000.0  # exclude illiquid names (tune as needed)

# --- Classification: percentile groups (Top 25% / Bottom 25%) ---
WINNER_PERCENTILE: float = 0.75
LOSER_PERCENTILE: float = 0.25

# --- Model / validation ---
TRAIN_TEST_SPLIT: float = 0.75
CV_FOLDS: int = 5


# --- Eulerpool API ---
EULERPOOL_API_KEY: str = os.environ.get("EULERPOOL_API_KEY", "")
EULERPOOL_BASE_URL: str = "https://api.eulerpool.com/api/1"

# --- Email Notifications ---
NOTIFY_EMAIL_TO: str = os.environ.get("NOTIFY_EMAIL_TO", "")
SMTP_HOST: str = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")


def get_period_by_label(label: str) -> tuple[str, str, str, str]:
    """Look up a classification period by its label (e.g. ``'Q1-2024'``)."""
    for p in CLASSIFICATION_PERIODS:
        if p[3] == label:
            return p
    raise KeyError(f"No classification period with label {label!r}")


def get_period_index(label: str) -> int:
    """Return the index of a classification period by its label."""
    for i, p in enumerate(CLASSIFICATION_PERIODS):
        if p[3] == label:
            return i
    raise KeyError(f"No classification period with label {label!r}")


def ensure_data_dir() -> Path:
    """Create the data cache directory if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
