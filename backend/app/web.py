"""Laptop browser page for manually exercising the public API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from starlette.responses import FileResponse


__all__ = ["router"]


router = APIRouter(include_in_schema=False)
_INDEX_PATH = Path(__file__).resolve().with_name("web") / "index.html"


@router.get("/", response_class=FileResponse)
async def laptop_page() -> FileResponse:
    """Serve the no-build laptop upload and results page."""
    return FileResponse(_INDEX_PATH, media_type="text/html")
