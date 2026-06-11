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
from urllib.parse import urlparse

import pandas as pd

from pipeline.logging_setup import get_logger
from pipeline.core_ats import extract_linkedin_handle, get_jobs_for_company as get_ats_jobs, AtsCache
from pipeline.url_validation import probe_urls_alive_batch

logger = get_logger(__name__)


# Local lookback window (days) — wide on purpose; local postings are sparse, and
# the seen-jobs tracker prevents the overlap from producing duplicate emails.
LOCAL_LOOKBACK_DAYS = 7

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
    # Lazy imports for the heavy / optional providers.
    from local_companies import ddg_search_for_jobs  # the provider-abstracted DDG helper

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
                        stats["ats_jobs"] += len(ats_jobs)
                        raw_jobs.extend(ats_jobs)
                except Exception as e:
                    logger.warning("ATS scrape failed for %s: %s", company_name, str(e)[:120])

            # DDG (LinkedIn posts + careers) via core_websearch (Google/DDG).
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

    # Ghost-listing HEAD-probe: drop dead DDG URLs before the AI sees them.
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
