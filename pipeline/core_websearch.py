"""
CORE WEB SEARCH MODULE
----------------------
A single, provider-agnostic `web_search(query, max_results)` used by the local
companies pipeline to find LinkedIn hiring posts + careers-page listings.

Why this exists: the local pipeline used to call DuckDuckGo (`ddgs`) directly,
which scrapes consumer search engines. From a GitHub Actions datacenter IP that
gets rate-limited / blocked hard (429 / 403 / connection timeouts) — in a real
run, 101 of ~114 company queries failed that way, yielding ~0 jobs.

The fix is to prefer Google's OFFICIAL Programmable Search JSON API when it's
configured. It's an authenticated API (not scraping), so it is NOT IP-blocked,
returns clean JSON, and has a free tier (100 queries/day). When the API keys
aren't set, we fall back to the old DDG path so nothing breaks and local dev
still works.

Provider order:
  1. Google Programmable Search  (if GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CX set)
  2. DuckDuckGo via `ddgs`        (fallback)

Both return the same normalized shape so callers don't care which ran:
    [{"title": str, "body": str, "href": str}, ...]

Env:
  GOOGLE_SEARCH_API_KEY   API key from a Google Cloud project with the
                          "Custom Search API" enabled.
  GOOGLE_SEARCH_CX        The Programmable Search Engine ID (the "cx" value).
                          Create one at programmablesearchengine.google.com and
                          set it to "Search the entire web".
"""

import os
from typing import List, Dict

import requests

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
REQUEST_TIMEOUT_SECONDS = 20


def google_search_configured() -> bool:
    """True when both Google Programmable Search credentials are present."""
    return bool(
        os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
        and os.environ.get("GOOGLE_SEARCH_CX", "").strip()
    )


def _google_search(query: str, max_results: int) -> List[Dict]:
    """Query Google Programmable Search. Returns normalized results or [] on failure.

    The free tier caps `num` at 10 per call; we never need more than a few, so a
    single call covers it. Raises nothing — logs and returns [] so the caller
    can fall back.
    """
    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "").strip()
    cx = os.environ.get("GOOGLE_SEARCH_CX", "").strip()
    try:
        resp = requests.get(
            GOOGLE_ENDPOINT,
            params={
                "key": api_key,
                "cx": cx,
                "q": query,
                "num": max(1, min(int(max_results), 10)),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        logger.warning("Google search transport error: %s", e)
        return []

    if resp.status_code == 429:
        logger.warning("Google search: daily quota exhausted (429) — falling back.")
        return []
    if resp.status_code != 200:
        logger.warning("Google search: HTTP %d %s", resp.status_code, (resp.text or "")[:160])
        return []

    try:
        items = resp.json().get("items", []) or []
    except ValueError:
        logger.warning("Google search: non-JSON response.")
        return []

    out = []
    for it in items:
        out.append({
            "title": it.get("title", "") or "",
            "body": it.get("snippet", "") or "",
            "href": it.get("link", "") or "",
        })
    return out


def _ddg_search(query: str, max_results: int, timelimit: str = "w") -> List[Dict]:
    """Fallback: DuckDuckGo via `ddgs`. Normalized to the same shape. [] on failure."""
    try:
        from ddgs import DDGS  # lazy — only imported on the fallback path
        results = DDGS().text(query, max_results=max_results, timelimit=timelimit)
        return [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in results
        ]
    except Exception as e:
        logger.warning("DDG search failed: %s", e)
        return []


def web_search(query: str, max_results: int = 3, timelimit: str = "w") -> List[Dict]:
    """Run a web search via the best available provider.

    Prefers Google Programmable Search (reliable, not IP-blocked) and falls back
    to DuckDuckGo when Google isn't configured OR returns nothing. Always returns
    a list of {"title", "body", "href"} dicts (possibly empty) — never raises.

    `timelimit` ("d"/"w"/"m") only applies to the DDG fallback; Google recency is
    handled by the caller re-verifying dates (LinkedIn activity IDs etc.).
    """
    if google_search_configured():
        results = _google_search(query, max_results)
        if results:
            return results
        # Google configured but returned nothing — try DDG as a second chance.
        return _ddg_search(query, max_results, timelimit)
    return _ddg_search(query, max_results, timelimit)
