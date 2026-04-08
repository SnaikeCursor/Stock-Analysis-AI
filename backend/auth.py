"""Anonymous user identification via ``X-User-ID`` (UUID v4)."""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db import UserPortfolio, UserProfile, get_session


def _parse_uuid(raw: str | None) -> uuid.UUID:
    if not raw or not raw.strip():
        raise HTTPException(
            status_code=401,
            detail="Missing X-User-ID header — generate a UUID in the client (localStorage).",
        )
    try:
        return uuid.UUID(raw.strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        ) from exc


async def get_or_create_user(
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> UserProfile:
    """Resolve ``UserProfile`` from ``X-User-ID``; create row + empty portfolio on first visit."""
    uid = _parse_uuid(request.headers.get("X-User-ID"))
    uuid_str = str(uid)

    result = await session.execute(
        select(UserProfile).where(UserProfile.uuid == uuid_str),
    )
    user = result.scalar_one_or_none()
    if user is None:
        user = UserProfile(uuid=uuid_str)
        session.add(user)
        await session.flush()
        pf = UserPortfolio(user_id=user.id, cash_balance=0.0)
        session.add(pf)
        await session.commit()
        await session.refresh(user)
    return user
