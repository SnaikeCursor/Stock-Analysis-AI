"""Parse and serialize signal ``portfolio_json`` text.

Legacy format: a JSON array of position dicts.
Current format: ``{"positions": [...], "requested_top_n": int}``.
"""

from __future__ import annotations

import json
from typing import Any


def serialize_portfolio_bundle(portfolio: list[dict[str, Any]], requested_top_n: int) -> str:
    """JSON for persistence; includes requested long count for UI / alerts."""
    return json.dumps(
        {"positions": portfolio, "requested_top_n": requested_top_n},
        indent=2,
    )


def parse_portfolio_json(raw: str | None) -> tuple[list[dict[str, Any]], int | None]:
    """Return (positions, requested_top_n or None for legacy payloads)."""
    if not raw:
        return [], None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [], None
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict):
        pos = data.get("positions")
        req = data.get("requested_top_n")
        if isinstance(pos, list):
            try:
                r = int(req) if req is not None else None
            except (TypeError, ValueError):
                r = None
            return pos, r
    return [], None
