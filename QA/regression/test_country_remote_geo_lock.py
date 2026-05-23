"""Regression: "UK or US (Remote)" location strings and "UK or US remote" description
phrases were not caught by the geo-lock filters, causing country-restricted remote roles
to reach the AI and the email. Fixed by:
  1. Adding _LOCATION_COUNTRY_PAREN_REMOTE_RE to catch "[country] (Remote)" in the
     location field.
  2. Adding description geo-lock templates for "[country] (Remote)" and
     "Remote — [country]" patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from pipeline.core_ai import quick_viability_check, _LOCATION_COUNTRY_PAREN_REMOTE_RE, _GEO_LOCK_REGEX


_LONG_DESC = (
    "We are looking for a software engineer to join our team. "
    "The role involves building Python backend services and REST APIs. "
    "You will work with FastAPI, PostgreSQL, and Docker on a daily basis. "
    "Strong Python skills are required along with experience in cloud platforms."
)  # 260+ chars — safely above the 150-char quick_viability_check minimum


def _row(location="", description=_LONG_DESC):
    return pd.Series({
        "title": "Software Engineer",
        "company": "Acme",
        "location": location,
        "description": description,
        "job_url": "https://example.com/job/1",
    })


# --- Location field regex unit tests ---

def test_location_country_paren_remote_uk():
    assert _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("UK (Remote)")


def test_location_country_paren_remote_uk_or_us():
    assert _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("UK or US (Remote)")


def test_location_country_paren_remote_united_kingdom():
    assert _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("United Kingdom (Remote)")


def test_location_country_paren_remote_slash_format():
    # US is not in the geo-lock alternation (handled separately), but the .{0,30}
    # middle allows any second country word after the first geo-locked one.
    assert _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("UK/US (Remote)")


def test_location_worldwide_remote_not_blocked():
    assert not _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("Remote")
    assert not _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("Worldwide (Remote)")
    assert not _LOCATION_COUNTRY_PAREN_REMOTE_RE.search("Remote — Anywhere")


# --- quick_viability_check integration tests ---

def test_uk_paren_remote_location_fails_viability():
    row = _row(location="UK (Remote)")
    viable, reason = quick_viability_check(row)
    assert not viable
    assert "geo-locked" in reason


def test_uk_or_us_paren_remote_location_fails_viability():
    row = _row(location="UK or US (Remote)")
    viable, reason = quick_viability_check(row)
    assert not viable
    assert "geo-locked" in reason


def test_plain_remote_location_passes():
    row = _row(location="Remote")
    viable, _ = quick_viability_check(row)
    assert viable


# --- Description geo-lock pattern tests (new templates) ---
# _GEO_LOCK_REGEX has no IGNORECASE flag — quick_viability_check lowercases the
# description before checking, so tests must also use lowercase.

def test_description_country_paren_remote_blocked():
    desc = ("this is a remote engineering role. uk (remote). "
            "requirements: python 3 years, django, rest apis. " * 5)
    assert _GEO_LOCK_REGEX.search(desc)


def test_description_remote_dash_country_blocked():
    # Use ASCII hyphen to avoid encoding issues with em/en dashes in test files.
    desc = ("software engineer position. remote - uk. "
            "experience with python and cloud required. " * 5)
    assert _GEO_LOCK_REGEX.search(desc)


def test_description_worldwide_remote_not_blocked():
    desc = ("we are a fully remote company hiring worldwide. "
            "no geographic restrictions. open to all locations. " * 3)
    assert not _GEO_LOCK_REGEX.search(desc)
