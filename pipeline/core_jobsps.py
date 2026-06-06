"""
CORE JOBS.PS MODULE
-------------------
Reads jobs from jobs.ps — the main general Palestinian job board — using the
JSON-LD `JobPosting` structured data each detail page embeds (the same schema
Google Jobs consumes). That's the gold-standard source: stable, machine-readable
title/company/location/description/dates, no fragile HTML scraping.

Flow:
  1. Fetch the listings page (https://www.jobs.ps/en/jobs) and pull the
     /en/jobs/<slug>-<id> detail URLs.
  2. Fetch each detail page, parse its <script type="application/ld+json">
     JobPosting block into a normalized job dict.

jobs.ps is a GENERAL board (heavy on NGO/admin/finance, some tech), so we pull a
healthy batch and let the existing filters + AI verdict separate tech from the
rest. Runs through the LOCAL filter path (Arabic-safe) like the other local
sources.

Output schema (matches the pipeline):
    {title, company, location, job_url, description, job_type, source, date_posted}

Never raises into the pipeline — a dead page or a malformed JSON-LD block is
logged and skipped so one bad listing can't stop the run.
"""

import json
import re
from datetime import datetime, timezone, timedelta
from html import unescape
from typing import List, Dict, Optional

import requests

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


BASE = "https://www.jobs.ps"
LISTINGS_URL = f"{BASE}/en/jobs"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
HTTP_TIMEOUT = 20

_DETAIL_RE = re.compile(r'href="(https?://www\.jobs\.ps/en/jobs/[a-z0-9\-]+-\d+)"', re.IGNORECASE)
_LDJSON_RE = re.compile(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>', re.S | re.IGNORECASE)


def extract_detail_urls(listings_html: str) -> List[str]:
    """Return the unique /en/jobs/<slug>-<id> detail URLs from a listings page."""
    seen = []
    for m in _DETAIL_RE.findall(listings_html or ""):
        if m not in seen:
            seen.append(m)
    return seen


def _clean_text(html_fragment: str) -> str:
    """Strip tags + unescape entities from a JSON-LD description. Keeps Arabic."""
    if not html_fragment:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", str(html_fragment), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _location_str(job_location) -> str:
    """Flatten the JSON-LD jobLocation (list/dict) into a readable string."""
    loc = job_location
    if isinstance(loc, list):
        loc = loc[0] if loc else None
    if isinstance(loc, dict):
        addr = loc.get("address")
        if isinstance(addr, dict):
            parts = [addr.get("addressLocality"), addr.get("addressRegion"),
                     addr.get("addressCountry")]
            joined = ", ".join(p for p in parts if p and isinstance(p, str))
            if joined:
                return joined
        if loc.get("name"):
            return str(loc["name"])
    return "Palestine"


def parse_jobposting_ld(detail_html: str, url: str, lookback_days: int,
                        now: Optional[datetime] = None) -> Optional[Dict]:
    """Parse one detail page's JSON-LD JobPosting into a job dict, or None.

    Returns None when there's no JobPosting block, it's malformed, or the post
    is older than the lookback window.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    posting = None
    for raw in _LDJSON_RE.findall(detail_html or ""):
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        # Could be a single object or a @graph list.
        candidates = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and isinstance(data.get("@graph"), list):
            candidates = data["@graph"]
        for c in candidates:
            if isinstance(c, dict) and c.get("@type") == "JobPosting":
                posting = c
                break
        if posting:
            break
    if not posting:
        return None

    # Recency from datePosted (date-only or full ISO).
    date_posted = posting.get("datePosted") or ""
    post_dt = None
    if date_posted:
        try:
            post_dt = datetime.fromisoformat(str(date_posted).replace("Z", "+00:00"))
            if post_dt.tzinfo is None:
                post_dt = post_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            post_dt = None
    if post_dt and post_dt < cutoff:
        return None

    title = str(posting.get("title", "")).strip()
    if not title:
        return None
    company = ""
    org = posting.get("hiringOrganization")
    if isinstance(org, dict):
        company = str(org.get("name", "")).strip()

    description = _clean_text(posting.get("description", ""))
    emp = posting.get("employmentType")
    job_type = "internship" if "intern" in title.lower() else "fulltime"
    if isinstance(emp, str) and "part" in emp.lower():
        job_type = "parttime"

    return {
        "title": title,
        "company": company,
        "location": _location_str(posting.get("jobLocation")),
        "job_url": url,
        "description": description[:6000],
        "job_type": job_type,
        "source": "jobs_ps",
        "date_posted": post_dt.isoformat() if post_dt else str(date_posted),
    }


def fetch_jobsps_jobs(lookback_days: int = 7, max_jobs: int = 40) -> List[Dict]:
    """Fetch recent jobs from jobs.ps. Never raises — returns [] on failure.

    Pulls the listings page, then up to `max_jobs` detail pages, parsing each
    one's JSON-LD JobPosting. The board is general (NGO-heavy), so `max_jobs`
    is generous to ensure enough tech roles reach the AI after filtering.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(LISTINGS_URL, headers=headers, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        logger.warning("jobs.ps listings fetch failed: %s", e)
        return []
    if resp.status_code != 200:
        logger.warning("jobs.ps listings: HTTP %d", resp.status_code)
        return []

    urls = extract_detail_urls(resp.text)[:max_jobs]
    logger.info("jobs.ps: %d detail page(s) to fetch.", len(urls))

    jobs: List[Dict] = []
    for url in urls:
        try:
            d = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            logger.warning("jobs.ps detail fetch failed (%s): %s", url[:60], e)
            continue
        if d.status_code != 200:
            continue
        job = parse_jobposting_ld(d.text, url, lookback_days)
        if job:
            jobs.append(job)

    logger.info("jobs.ps: %d job(s) parsed within %dd.", len(jobs), lookback_days)
    return jobs
