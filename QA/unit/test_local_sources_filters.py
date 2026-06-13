"""core_local_sources freshness + noise filters (local-market quality).

Locks two fixes:
  * drop_stale_ats_jobs — ATS APIs return forgotten-but-still-200 listings
    (e.g. an Innotic role left up for months); we age them out by date_posted.
    Non-ATS rows and rows with no parseable date are KEPT.
  * is_linkedin_only_website — companies whose only 'website' is a LinkedIn URL
    (e.g. Aurora Technologies) yield only DDG search noise; we detect them so
    the caller can skip the per-company DDG dance.

Pure functions, no network. Dates are built relative to now so the test never
goes stale itself.
"""

from datetime import datetime, timezone, timedelta

from pipeline.core_local_sources import (
    drop_stale_ats_jobs,
    is_linkedin_only_website,
    _parse_job_date,
    ATS_MAX_AGE_DAYS,
)


def _iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


# --- is_linkedin_only_website ----------------------------------------------

def test_linkedin_company_url_is_flagged():
    assert is_linkedin_only_website("https://www.linkedin.com/company/auroratech-ps/")


def test_real_careers_site_is_not_flagged():
    assert not is_linkedin_only_website("https://www.example-tech.ps/careers")


def test_blank_and_nan_website_not_flagged():
    assert not is_linkedin_only_website("")
    assert not is_linkedin_only_website("nan")
    assert not is_linkedin_only_website(None)


# --- _parse_job_date --------------------------------------------------------

def test_parse_iso_with_z_suffix():
    dt = _parse_job_date("2026-01-15T10:00:00Z")
    assert dt is not None and dt.tzinfo is not None


def test_parse_date_only():
    dt = _parse_job_date("2026-01-15")
    assert dt is not None and dt.year == 2026


def test_parse_unparseable_returns_none():
    assert _parse_job_date("") is None
    assert _parse_job_date("not a date") is None
    assert _parse_job_date(None) is None


# --- drop_stale_ats_jobs ----------------------------------------------------

def test_drops_old_ats_job():
    jobs = [{"source": "ats_api", "job_url": "u1", "date_posted": _iso(ATS_MAX_AGE_DAYS + 30)}]
    assert drop_stale_ats_jobs(jobs) == []


def test_keeps_fresh_ats_job():
    jobs = [{"source": "ats_api", "job_url": "u1", "date_posted": _iso(5)}]
    assert len(drop_stale_ats_jobs(jobs)) == 1


def test_keeps_ats_job_with_no_date():
    # Unknown date → keep (never drop on uncertainty).
    jobs = [{"source": "ats_api", "job_url": "u1", "date_posted": ""}]
    assert len(drop_stale_ats_jobs(jobs)) == 1


def test_does_not_touch_non_ats_sources():
    # A very old DDG row is left for the dead-URL probe, not aged out here.
    jobs = [{"source": "ddg_linkedin", "job_url": "u1", "date_posted": _iso(999)}]
    assert len(drop_stale_ats_jobs(jobs)) == 1


def test_mixed_batch_only_stale_ats_removed():
    jobs = [
        {"source": "ats_api", "job_url": "old", "date_posted": _iso(ATS_MAX_AGE_DAYS + 1)},
        {"source": "ats_api", "job_url": "new", "date_posted": _iso(1)},
        {"source": "ddg_website", "job_url": "ddg", "date_posted": _iso(999)},
    ]
    kept = {j["job_url"] for j in drop_stale_ats_jobs(jobs)}
    assert kept == {"new", "ddg"}
