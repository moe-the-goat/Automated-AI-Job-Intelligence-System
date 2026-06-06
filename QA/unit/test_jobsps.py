"""core_jobsps parser tests (jobs.ps JSON-LD JobPosting).

Locks the pure parsers: detail-URL extraction, JSON-LD JobPosting parsing,
location flattening, recency window, and graceful handling of missing/malformed
blocks. No network — synthetic HTML fed straight to the parsers.
"""
import json
from datetime import datetime, timezone, timedelta

from pipeline.core_jobsps import (
    extract_detail_urls,
    parse_jobposting_ld,
    _location_str,
)

_NOW = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)


def _ld(posting: dict) -> str:
    return f'<script type="application/ld+json">{json.dumps(posting)}</script>'


def _posting(**over):
    p = {
        "@context": "https://schema.org",
        "@type": "JobPosting",
        "title": "Software Engineer",
        "description": "Build <b>great</b> software.<br>Python required.",
        "datePosted": "2026-06-05",
        "hiringOrganization": {"@type": "Organization", "name": "ASAL Technologies"},
        "jobLocation": [{"@type": "Place", "address": {
            "@type": "PostalAddress", "addressLocality": "Ramallah",
            "addressCountry": "Palestine"}}],
        "employmentType": "Full time",
    }
    p.update(over)
    return p


# ---------------------------------------------------------------------------
# extract_detail_urls
# ---------------------------------------------------------------------------

def test_extract_detail_urls_dedups_and_filters():
    html = (
        '<a href="https://www.jobs.ps/en/jobs/social-worker-67158">x</a>'
        '<a href="https://www.jobs.ps/en/jobs/site-engineer-67152">y</a>'
        '<a href="https://www.jobs.ps/en/jobs/social-worker-67158">dup</a>'
        '<a href="https://www.jobs.ps/en/tenders/rfq-1-2-3.html">tender</a>'
        '<a href="https://www.jobs.ps/en/employer/login">login</a>'
    )
    urls = extract_detail_urls(html)
    assert urls == [
        "https://www.jobs.ps/en/jobs/social-worker-67158",
        "https://www.jobs.ps/en/jobs/site-engineer-67152",
    ]


def test_extract_detail_urls_empty():
    assert extract_detail_urls("") == []
    assert extract_detail_urls("<a href='/en/tenders/x.html'>t</a>") == []


# ---------------------------------------------------------------------------
# _location_str
# ---------------------------------------------------------------------------

def test_location_from_address_list():
    loc = [{"@type": "Place", "address": {"addressLocality": "Gaza Strip",
            "addressCountry": "Palestine"}}]
    assert _location_str(loc) == "Gaza Strip, Palestine"


def test_location_fallback_to_name_then_default():
    assert _location_str({"name": "Remote"}) == "Remote"
    assert _location_str(None) == "Palestine"


# ---------------------------------------------------------------------------
# parse_jobposting_ld
# ---------------------------------------------------------------------------

def test_parses_full_posting():
    html = _ld(_posting())
    job = parse_jobposting_ld(html, "https://www.jobs.ps/en/jobs/software-engineer-1", 7, now=_NOW)
    assert job is not None
    assert job["title"] == "Software Engineer"
    assert job["company"] == "ASAL Technologies"
    assert job["location"] == "Ramallah, Palestine"
    assert job["source"] == "jobs_ps"
    assert job["job_type"] == "fulltime"
    assert "Python required." in job["description"]
    assert "<b>" not in job["description"]   # tags stripped


def test_internship_title_sets_job_type():
    html = _ld(_posting(title="Backend Developer Internship"))
    job = parse_jobposting_ld(html, "u", 7, now=_NOW)
    assert job["job_type"] == "internship"


def test_drops_old_posting():
    html = _ld(_posting(datePosted="2026-04-01"))
    assert parse_jobposting_ld(html, "u", 7, now=_NOW) is None


def test_handles_graph_wrapper():
    graph = {"@context": "https://schema.org", "@graph": [
        {"@type": "WebSite"}, _posting(title="Data Engineer")]}
    html = f'<script type="application/ld+json">{json.dumps(graph)}</script>'
    job = parse_jobposting_ld(html, "u", 7, now=_NOW)
    assert job is not None and job["title"] == "Data Engineer"


def test_no_jobposting_returns_none():
    html = '<script type="application/ld+json">{"@type":"WebSite"}</script>'
    assert parse_jobposting_ld(html, "u", 7, now=_NOW) is None


def test_malformed_jsonld_returns_none():
    html = '<script type="application/ld+json">{not valid json</script>'
    assert parse_jobposting_ld(html, "u", 7, now=_NOW) is None


def test_missing_title_returns_none():
    html = _ld(_posting(title=""))
    assert parse_jobposting_ld(html, "u", 7, now=_NOW) is None


def test_arabic_company_preserved():
    html = _ld(_posting(hiringOrganization={"name": "جمعية غزة للزراعة"}))
    job = parse_jobposting_ld(html, "u", 7, now=_NOW)
    assert job["company"] == "جمعية غزة للزراعة"
