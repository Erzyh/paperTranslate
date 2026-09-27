"""Semantic Scholar Graph API client for reference details lookup.

Isolated in its own module so the httpx transport can be injected in tests
(httpx.MockTransport); the API layer never builds the client itself. The
fetch function never raises: any failure (network error, timeout, HTTP
status, malformed JSON, empty result) collapses into the contract's
{"found": False, ...} payload.
"""
from __future__ import annotations

import os

import httpx

# Base URL is a module constant so tests can point it at a .invalid host.
API_BASE_URL = "https://api.semanticscholar.org"

# Optional API key: the anonymous shared pool is aggressively rate-limited
# (HTTP 429); a free key from semanticscholar.org lifts that. Read per call
# so the env var can be set without a restart-order dependency in tests.
_API_KEY_ENV = "PAPERTRANSLATE_S2_API_KEY"
_SEARCH_PATH = "/graph/v1/paper/search"
_FIELDS = "title,authors,year,abstract,url"
_TIMEOUT = 8.0

# Test hook: inject an httpx.MockTransport so lookups never touch the real
# Semantic Scholar API in tests. None selects the default transport. A
# transport passed directly to fetch_paper_details takes precedence.
TRANSPORT: httpx.AsyncBaseTransport | None = None


def not_found() -> dict:
    """Fresh failure payload per the contract (found=false, all fields empty)."""
    return {"found": False, "title": None, "authors": [], "year": None,
            "abstract": None, "url": None}


def _parse_paper(data: dict) -> dict:
    """Map one search response to the contract payload (found=false if empty)."""
    papers = data.get("data") or []
    if not isinstance(papers, list) or not papers:
        return not_found()
    paper = papers[0]
    if not isinstance(paper, dict):
        return not_found()
    authors = [a.get("name") for a in paper.get("authors") or []
               if isinstance(a, dict) and a.get("name")]
    year = paper.get("year")
    return {
        "found": True,
        "title": paper.get("title"),
        "authors": authors,
        "year": year if isinstance(year, int) and not isinstance(year, bool)
        else None,
        "abstract": paper.get("abstract"),
        "url": paper.get("url"),
    }


async def fetch_paper_details(
    query: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Search Semantic Scholar for one paper matching the query text.

    Returns the reference-details contract payload; never raises. Only a
    parsed paper yields found=true — the caller caches exactly those.
    """
    if transport is None:
        transport = TRANSPORT
    api_key = os.environ.get(_API_KEY_ENV, "").strip()
    headers = {"x-api-key": api_key} if api_key else None
    try:
        async with httpx.AsyncClient(
            base_url=API_BASE_URL,
            timeout=_TIMEOUT,
            transport=transport,
            headers=headers,
        ) as client:
            response = await client.get(
                _SEARCH_PATH,
                params={"query": query, "fields": _FIELDS, "limit": 1},
            )
            response.raise_for_status()
            return _parse_paper(response.json())
    except Exception:  # noqa: BLE001 - contract: any failure is found=false
        return not_found()
