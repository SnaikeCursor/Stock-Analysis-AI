#!/usr/bin/env python3
"""Validate Eulerpool API coverage for Swiss stocks.

Tests:
1. Profile + income statement for a handful of active SPI-Extra tickers
2. Quarterly fundamentals availability & historical depth
3. Delisted ticker coverage (CSGN.SW = Credit Suisse)
4. Index constituents (SPI, SPI Extra if available)
5. Point-in-time endpoints

Usage:
    export EULERPOOL_API_KEY=your_key
    python test_eulerpool.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eulerpool_client import EulerpoolClient

TEST_TICKERS = [
    "LOGN.SW",   # Logitech
    "VACN.SW",   # VAT Group
    "LONN.SW",   # Lonza
    "TEMN.SW",   # Temenos
    "SREN.SW",   # Swiss Re
]
DELISTED_TICKERS = [
    "CSGN.SW",   # Credit Suisse (delisted 2023)
]
CANDIDATE_INDICES = [
    "spi",
    "spiextra",
    "spi-extra",
    "smim",
    "smi",
]


def _pp(label: str, data, max_items: int = 3):
    if data is None:
        print(f"  {label}: ⚠️  None (not available)")
        return
    if isinstance(data, list):
        print(f"  {label}: {len(data)} records")
        for item in data[:max_items]:
            print(f"    → {json.dumps(item, default=str)[:200]}")
    elif isinstance(data, dict):
        keys = list(data.keys())[:15]
        print(f"  {label}: dict with keys {keys}")
    else:
        print(f"  {label}: {str(data)[:200]}")


def main():
    print("=" * 72)
    print("  EULERPOOL API VALIDATION — Swiss Stocks")
    print("=" * 72)

    try:
        client = EulerpoolClient()
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("Set EULERPOOL_API_KEY environment variable and retry.")
        sys.exit(1)

    total_requests = 0
    passed = 0
    failed = 0

    # ── Test 1: Active ticker profiles ───────────────────────────────────
    print("\n--- Test 1: Active ticker profiles ---")
    for ticker in TEST_TICKERS:
        print(f"\n  [{ticker}]")
        profile = client.profile(ticker)
        total_requests += 1
        if profile:
            name = profile.get("name", "?")
            sector = profile.get("sector", "?")
            exchange = profile.get("exchange", "?")
            print(f"    Profile: {name} | {sector} | {exchange}")
            passed += 1
        else:
            print(f"    Profile: ⚠️  NOT FOUND")
            failed += 1

    # ── Test 2: Income statements (annual + quarterly) ───────────────────
    print("\n--- Test 2: Income statements ---")
    for ticker in TEST_TICKERS[:3]:
        print(f"\n  [{ticker}]")

        annual = client.income_statement(ticker)
        total_requests += 1
        if annual:
            years = [r.get("period", "?")[:4] for r in annual[:5]]
            print(f"    Annual: {len(annual)} periods, earliest years: {years[-3:]}")
            passed += 1
        else:
            print(f"    Annual: ⚠️  NOT FOUND")
            failed += 1

        quarterly = client.income_statement_quarterly(ticker)
        total_requests += 1
        if quarterly:
            periods = [r.get("period", "?")[:10] for r in quarterly[:5]]
            print(f"    Quarterly: {len(quarterly)} periods, recent: {periods[:3]}")
            passed += 1
        else:
            print(f"    Quarterly: ⚠️  NOT FOUND")
            failed += 1

    # ── Test 3: Quarterly fundamentals ───────────────────────────────────
    print("\n--- Test 3: Quarterly fundamentals (consolidated) ---")
    for ticker in TEST_TICKERS[:2]:
        print(f"\n  [{ticker}]")
        fq = client.fundamentals_quarterly(ticker)
        total_requests += 1
        _pp("Fundamentals-Q", fq)
        if fq:
            passed += 1
        else:
            failed += 1

    # ── Test 4: Metrics / Ratios ─────────────────────────────────────────
    print("\n--- Test 4: Financial metrics & ratios ---")
    for ticker in TEST_TICKERS[:2]:
        print(f"\n  [{ticker}]")
        m = client.metrics(ticker)
        total_requests += 1
        _pp("Metrics", m)
        if m:
            passed += 1
        else:
            failed += 1

    # ── Test 5: Delisted ticker ──────────────────────────────────────────
    print("\n--- Test 5: Delisted tickers ---")
    for ticker in DELISTED_TICKERS:
        print(f"\n  [{ticker}] (Credit Suisse, delisted 2023)")
        profile = client.profile(ticker)
        total_requests += 1
        if profile:
            print(f"    Profile: {profile.get('name', '?')} ✅ (delisted covered!)")
            passed += 1
        else:
            print(f"    Profile: ⚠️  NOT FOUND")
            failed += 1

        inc = client.income_statement(ticker)
        total_requests += 1
        if inc:
            print(f"    Income: {len(inc)} periods ✅")
            passed += 1
        else:
            print(f"    Income: ⚠️  NOT FOUND")
            failed += 1

    # ── Test 6: Point-in-time endpoints ──────────────────────────────────
    print("\n--- Test 6: Point-in-time endpoints ---")
    ticker = TEST_TICKERS[0]
    print(f"\n  [{ticker}]")

    pit_est = client.pit_estimates(ticker)
    total_requests += 1
    _pp("PIT Estimates", pit_est)
    if pit_est:
        passed += 1
    else:
        failed += 1

    pit_prof = client.pit_profile(ticker)
    total_requests += 1
    _pp("PIT Profile", pit_prof)
    if pit_prof:
        passed += 1
    else:
        failed += 1

    # ── Test 7: Index constituents ───────────────────────────────────────
    print("\n--- Test 7: Index constituents ---")
    for idx in CANDIDATE_INDICES:
        constituents = client.index_constituents(idx)
        total_requests += 1
        if constituents:
            print(f"  [{idx}]: {len(constituents)} members ✅")
            sample = [c.get("ticker", "?") for c in constituents[:5]]
            print(f"    Sample: {sample}")
            passed += 1
        else:
            print(f"  [{idx}]: ⚠️  NOT FOUND")
            failed += 1

    # ── Test 8: Valuation history ────────────────────────────────────────
    print("\n--- Test 8: Valuation history ---")
    ticker = TEST_TICKERS[0]
    print(f"\n  [{ticker}]")
    vh = client.valuation_history(ticker)
    total_requests += 1
    _pp("Valuation History", vh)
    if vh:
        passed += 1
    else:
        failed += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"  API requests used: {total_requests}")
    print(f"  (Free tier allows 1,000/month)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
