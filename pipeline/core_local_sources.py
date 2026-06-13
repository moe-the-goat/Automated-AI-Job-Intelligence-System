"""
CORE LOCAL SOURCES MODULE
-------------------------
Reusable collection of the Palestinian local-market job sources, extracted so
BOTH the single-user `local_companies.py` and the multi-user `multi_user_runner.py`
can pull local jobs from one shared implementation (no duplication, fixes apply
to both at once).

`collect_local_raw_jobs()` runs every local source and returns a flat list of
raw job dicts (un-filtered, un-AI'd) tagged with `source`:
    * Per-company ATS  (Greenhouse/Lever/Ashby/Workable/Workday + Jina fallback)
    * Per-company DDG  (LinkedIn posts + careers pages, via core_websearch)
    * Per-company JobSpy (LinkedIn, name-matched)
    * Public Telegram channels  (t.me/s web preview)
    * jobs.ps  (JSON-LD JobPosting)
Then runs the ghost-listing HEAD-probe to drop dead DDG URLs.

The CALLER is responsible for: pipeline filtering (apply_pipeline_filters with
local=True), AI evaluation, dedup tracker, and delivery. This module only
GATHERS the raw local jobs.
"""

import os
import re
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse

import pandas as pd

from pipeline.logging_setup import get_logger
from pipeline.core_ats import extract_linkedin_handle, get_jobs_for_company as get_ats_jobs, AtsCache
from pipeline.url_validation import probe_urls_alive_batch, is_specific_job_url_like

logger = get_logger(__name__)


# Local lookback window (days) — wide on purpose; local postings are sparse, and
# the seen-jobs tracker prevents the overlap from producing duplicate emails.
LOCAL_LOOKBACK_DAYS = 7

# ATS APIs (Greenhouse/Lever/…) return EVERY open posting, including ones a
# company forgot to close — these still answer 200 so the dead-URL probe can't
# catch them. Drop ATS jobs whose date_posted is older than this so a stale
# listing (e.g. an Innotic role left up for months) never reaches the AI/email.
# A real local role rarely stays genuinely open past two months.
ATS_MAX_AGE_DAYS = 60

# Public Telegram job channels (read via t.me/s/<handle> — no login/token).
TELEGRAM_JOB_CHANNELS = [
    "fromcodetocareer",
]

# The Excel company lists at the repo root.
LOCAL_COMPANY_FILES = [
    "IT Companies - Nablus.xlsx",
    "IT Companies - Ramallah.xlsx",
]


def extract_domain(url):
    """Extract the base domain from a URL (https://www.company.com/jobs -> company.com)."""
    if pd.isna(url) or not str(url).strip():
        return ""
    try:
        domain = urlparse(str(url)).netloc
        return domain.replace("www.", "")
    except Exception:
        return ""


# LinkedIn activity IDs are snowflake-like: the top ~41 bits encode a UNIX
# timestamp in MILLISECONDS. Right-shifting the 64-bit ID by 22 strips the low
# sequence bits and yields a millisecond timestamp.
# Verified: 7397959444342575104 >> 22 = 1763830509824 ms -> 2025-11-22 UTC.
_LINKEDIN_ACTIVITY_RE = re.compile(r"activity-(\d+)")
_LINKEDIN_HANDLE_RE = re.compile(r"linkedin\.com/posts/([a-z0-9\-]+?)(?:_|/)")


def linkedin_post_date(url):
    """Decode the post date from a LinkedIn activity URL. None if undecodable.

    BUG HISTORY: an earlier version treated the shifted value as seconds, which
    overflowed fromtimestamp() and made this always return None — so the date
    filter never fired and months-old posts slipped through. Fixed by /1000.
    """
    match = _LINKEDIN_ACTIVITY_RE.search(url or "")
    if not match:
        return None
    try:
        activity_id = int(match.group(1))
        ts_milliseconds = activity_id >> 22
        return datetime.fromtimestamp(ts_milliseconds / 1000, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def linkedin_handle_matches(url, company_name):
    """True if the post URL's handle contains a meaningful chunk of the company
    name. Prevents DDG false positives where a generic first word (e.g.
    'Future') matches a totally unrelated post.
    """
    match = _LINKEDIN_HANDLE_RE.search((url or "").lower())
    if not match:
        return False
    handle = match.group(1)
    name_tokens = re.findall(r"[a-z]+", company_name.lower())
    return any(len(tok) >= 3 and tok in handle for tok in name_tokens)


def ddg_search_for_jobs(company_name, domain, linkedin_handle=None):
    """Search for recent job posts on LinkedIn and the company website.

    Uses `web_search` (Google Programmable Search when configured, else
    DuckDuckGo) so the local pipeline isn't reliant on IP-blockable scraping.
    When `linkedin_handle` is provided, the LinkedIn query is the precise
    site:linkedin.com/company/{handle}/posts form.

    Moved here from local_companies.py (2026-06-14 cleanup) so the multi-user
    path no longer reaches into the legacy single-user entry point.
    """
    from pipeline.core_websearch import web_search
    jobs_found = []

    short_name = company_name.split()[0] if len(company_name.split()) > 0 else company_name

    # 1. LinkedIn Posts (handle-precise when available, else name search).
    try:
        if linkedin_handle:
            q1 = f'site:linkedin.com/company/{linkedin_handle}/posts (hiring OR vacancy OR "looking for" OR job)'
            logger.info("Searching LinkedIn Posts (handle '%s') for: %s...", linkedin_handle, company_name)
        else:
            q1 = f'site:linkedin.com/posts {short_name} (hiring OR vacancy OR "looking for" OR job)'
            logger.info("Searching LinkedIn Posts for: %s (using '%s')...", company_name, short_name)
        res1 = web_search(q1, max_results=3, timelimit="w")  # past week (DDG fallback only)
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
        for r in res1:
            title = r.get("title", "")
            body = r.get("body", "")
            link = r.get("href", "")

            # Require a REAL, DATEABLE post: no activity id -> not a post (skip);
            # older than the lookback -> skip.
            post_date = linkedin_post_date(link)
            if post_date is None:
                logger.info("Skipping non-post LinkedIn URL for %s: %s", company_name, link[:80])
                continue
            if post_date < cutoff:
                logger.info("Skipping old post for %s: posted %s", company_name, post_date.date())
                continue
            if not linkedin_handle_matches(link, company_name):
                logger.info("Skipping unrelated post for %s: handle mismatch (%s)", company_name, link[:80])
                continue

            jobs_found.append({
                "title": title[:120] if title else f"{company_name} — LinkedIn hiring post",
                "company": company_name,
                "location": "Local/Remote",
                "job_url": link,
                "description": body,
                "job_type": "fulltime",
                "source": "ddg_linkedin",
            })
    except Exception as e:
        logger.warning("DDG LinkedIn search failed for %s: %s", company_name, e)

    # 2. Company Website (require a SPECIFIC job-detail URL, not a landing page).
    if domain:
        try:
            q2 = f"site:{domain} (hiring OR careers OR jobs OR vacancy)"
            logger.info("Searching Website (%s) for: %s...", domain, company_name)
            res2 = web_search(q2, max_results=3, timelimit="w")
            for r in res2:
                title = r.get("title", "")
                body = r.get("body", "")
                link = r.get("href", "")
                if not is_specific_job_url_like(link):
                    logger.info("Skipping non-specific URL for %s: %s", company_name, link[:80])
                    continue
                jobs_found.append({
                    "title": title[:120] if title else f"{company_name} — careers page listing",
                    "company": company_name,
                    "location": "Local/Remote",
                    "job_url": link,
                    "description": body,
                    "job_type": "fulltime",
                    "source": "ddg_website",
                })
        except Exception as e:
            logger.warning("DDG Website search failed for %s: %s", company_name, e)

    time.sleep(1)  # be nice to the search API
    return jobs_found


def is_linkedin_only_website(url):
    """True when a company's 'jobs website' is just a LinkedIn URL, not a real
    careers page.

    Entries like "Aurora Technologies" list a linkedin.com/company/... URL as
    their jobs website and have no scrapeable careers page. ATS detection fails,
    and the per-company DDG/website dance can then only ever return old or
    non-specific LinkedIn URLs — pure search noise (and captcha/429 hits) with
    zero usable jobs. We skip that dance for these. ATS + JobSpy still run.
    """
    s = str(url or "").strip().lower()
    if not s or s == "nan":
        return False
    return "linkedin.com" in extract_domain(s)


def _parse_job_date(raw):
    """Best-effort parse of an ATS date_posted into an aware UTC datetime.

    ATS fetchers emit a mix of ISO-8601 (Greenhouse updated_at, Ashby,
    Workable) and date-only strings. Returns None when unparseable — the
    caller treats None as "unknown, keep" so a missing/odd date never silently
    drops a real job.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    # Normalize a trailing Z to +00:00 for fromisoformat.
    iso = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        # Fall back to a plain date (YYYY-MM-DD) prefix.
        try:
            dt = datetime.fromisoformat(s[:10])
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def drop_stale_ats_jobs(raw_jobs, max_age_days=ATS_MAX_AGE_DAYS):
    """Drop ATS-sourced jobs older than max_age_days by their date_posted.

    Only ATS rows (source startswith 'ats_') are aged out — they come from APIs
    that return every open posting including forgotten ones. A row with no
    parseable date is KEPT (we never drop on uncertainty). Other sources
    (ddg/jobspy/telegram/jobs.ps) are recency-filtered upstream and untouched.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_days * 86400
    before = len(raw_jobs)
    kept = []
    dropped = 0
    for j in raw_jobs:
        if str(j.get("source", "")).startswith("ats_"):
            dt = _parse_job_date(j.get("date_posted"))
            if dt is not None and dt.timestamp() < cutoff:
                dropped += 1
                continue
        kept.append(j)
    if dropped:
        logger.info(
            "Stale-ATS filter: dropped %d job(s) older than %d days (of %d).",
            dropped, max_age_days, before,
        )
    return kept


def load_local_companies():
    """Load all companies from the Excel files, normalizing column casing.

    The two sheets differ in casing (`LinkedIn Profile` vs `LinkedIn profile`,
    `Jobs Website` vs `Jobs website`), so we lower-case the headers.
    """
    dfs = []
    for f in LOCAL_COMPANY_FILES:
        if os.path.exists(f):
            try:
                df = pd.read_excel(f)
                df.columns = [str(c).strip().lower() for c in df.columns]
                dfs.append(df)
            except Exception as e:
                logger.warning("Error loading %s: %s", f, e)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def collect_local_raw_jobs(*, gemini_key=None, lookback_days=LOCAL_LOOKBACK_DAYS):
    """Gather raw local-market jobs from every local source. Returns a list of dicts.

    Pure gathering — no filtering, AI, or delivery. Never raises: each source is
    individually try/except'd so one dead source can't stop the rest. `gemini_key`
    enables the ATS Jina-extraction fallback (optional).

    Returns (raw_jobs, stats) where stats has per-source counts for logging.
    """
    raw_jobs = []
    stats = {"ats_jobs": 0, "ddg_jobs": 0, "jobspy_jobs": 0, "telegram_jobs": 0, "jobsps_jobs": 0}

    companies_df = load_local_companies()
    if companies_df.empty:
        logger.warning("No local companies loaded — skipping per-company sources.")
    else:
        ats_cache = AtsCache()
        for _, row in companies_df.iterrows():
            company_name = str(row.get("company name", "")).strip()
            website = str(row.get("jobs website", ""))
            linkedin_url = str(row.get("linkedin profile", ""))
            domain = extract_domain(website)
            linkedin_handle = (
                extract_linkedin_handle(linkedin_url)
                if linkedin_url and linkedin_url.lower() != "nan" else None
            )
            if not company_name or company_name == "nan":
                continue

            # ATS API (cached detection) + Jina fallback when a Gemini key is set.
            if website and website.lower() != "nan":
                try:
                    ats_jobs = get_ats_jobs(
                        company_name, website, cache=ats_cache,
                        gemini_api_key=gemini_key, jina_fallback=bool(gemini_key),
                    )
                    if ats_jobs:
                        logger.info("ATS yielded %d job(s) for %s", len(ats_jobs), company_name)
                        # Tag the source so the stale-ATS filter can target these
                        # (core_ats doesn't set a source field itself).
                        for j in ats_jobs:
                            j.setdefault("source", "ats_api")
                        stats["ats_jobs"] += len(ats_jobs)
                        raw_jobs.extend(ats_jobs)
                except Exception as e:
                    logger.warning("ATS scrape failed for %s: %s", company_name, str(e)[:120])

            # DDG (LinkedIn posts + careers) via core_websearch (Google/DDG).
            # Skip companies whose only 'website' is a LinkedIn URL — ATS can't
            # detect them and the DDG dance returns only old / non-specific links
            # (pure noise + captcha hits) for zero usable jobs.
            if is_linkedin_only_website(website):
                logger.info(
                    "Skipping per-company DDG for %s — website is LinkedIn-only (no careers page).",
                    company_name,
                )
            else:
                try:
                    ddg_jobs = ddg_search_for_jobs(company_name, domain, linkedin_handle=linkedin_handle)
                    stats["ddg_jobs"] += len(ddg_jobs)
                    raw_jobs.extend(ddg_jobs)
                except Exception as e:
                    logger.warning("DDG search failed for %s: %s", company_name, str(e)[:120])

            # JobSpy (LinkedIn, name-matched to avoid generic results).
            try:
                from jobspy import scrape_jobs  # lazy import
                jobspy_res = scrape_jobs(
                    site_name=["linkedin"], search_term=company_name,
                    location="State of Palestine", distance=100, results_wanted=5,
                    hours_old=lookback_days * 24,
                )
                for _, j_row in jobspy_res.iterrows():
                    found_company = str(j_row.get("company", "")).lower()
                    if company_name.lower() in found_company or found_company in company_name.lower():
                        raw_jobs.append(j_row.to_dict())
                        stats["jobspy_jobs"] += 1
            except Exception as e:
                logger.warning("JobSpy failed for %s: %s", company_name, str(e)[:120])

        ats_cache.save()

    # Public Telegram channels.
    if TELEGRAM_JOB_CHANNELS:
        try:
            from pipeline.core_telegram import fetch_telegram_jobs
            tg_jobs = fetch_telegram_jobs(TELEGRAM_JOB_CHANNELS, lookback_days=lookback_days)
            if tg_jobs:
                logger.info("Telegram channels contributed %d post(s).", len(tg_jobs))
                stats["telegram_jobs"] = len(tg_jobs)
                raw_jobs.extend(tg_jobs)
        except Exception as e:
            logger.warning("Telegram sweep failed: %s", e)

    # jobs.ps (JSON-LD).
    try:
        from pipeline.core_jobsps import fetch_jobsps_jobs
        jobsps = fetch_jobsps_jobs(lookback_days=lookback_days)
        if jobsps:
            logger.info("jobs.ps contributed %d job(s).", len(jobsps))
            stats["jobsps_jobs"] = len(jobsps)
            raw_jobs.extend(jobsps)
    except Exception as e:
        logger.warning("jobs.ps sweep failed: %s", e)

    # Drop stale ATS listings (still 200 in the API but long abandoned) and
    # dead DDG URLs (404/410) before the AI sees them.
    raw_jobs = drop_stale_ats_jobs(raw_jobs)
    raw_jobs = drop_dead_ddg_urls(raw_jobs)
    return raw_jobs, stats


def drop_dead_ddg_urls(raw_jobs):
    """HEAD-probe DDG-sourced URLs and drop the dead ones. Returns filtered list.

    Only DDG rows (source startswith 'ddg_') are probed — ATS/JobSpy/Telegram/
    jobs.ps come from live endpoints or are already recency-filtered. Unprobed
    entries default to kept.
    """
    def _is_ddg(j):
        return str(j.get("source", "")).startswith("ddg_")

    ddg_urls = [j.get("job_url") for j in raw_jobs if _is_ddg(j) and j.get("job_url")]
    if not ddg_urls:
        return raw_jobs

    unique_urls = list(set(ddg_urls))
    logger.info("Verifying %d DDG-sourced URL(s) via HEAD probe...", len(unique_urls))
    alive_map = probe_urls_alive_batch(unique_urls)
    before = len(raw_jobs)
    kept = [
        j for j in raw_jobs
        if not _is_ddg(j) or alive_map.get(j.get("job_url"), True)
    ]
    dropped = before - len(kept)
    if dropped:
        logger.info("Ghost-listing filter: dropped %d dead URL(s).", dropped)
    return kept
