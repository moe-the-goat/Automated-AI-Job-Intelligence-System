"""LinkedIn helpers in local_companies.py.

The post-date decoder once had a ms-vs-s bug that let 5-month and 1-year
old posts slip into the email. The regression for that lives in
regression/test_date_decoder_ms_bug.py — this file is the everyday coverage.
"""
from datetime import datetime, timezone, timedelta
from pipeline.core_local_sources import linkedin_post_date, linkedin_handle_matches, LOCAL_LOOKBACK_DAYS


def test_post_date_decodes_iconnect_url():
    url = "https://www.linkedin.com/posts/iconnect-tech_hiring-fullstackdeveloper-ai-activity-7397959444342575104-p6DK"
    d = linkedin_post_date(url)
    assert d is not None
    assert d.year == 2025
    assert d.month == 11  # November 2025


def test_post_date_decodes_smartweb_url():
    url = "https://www.linkedin.com/posts/smartweb-labs_hiring-python-python-activity-7284632728031944704-93L4"
    d = linkedin_post_date(url)
    assert d is not None
    assert d.year == 2025
    assert d.month == 1   # January 2025


def test_post_date_returns_none_for_url_without_activity_id():
    assert linkedin_post_date("https://www.linkedin.com/posts/someone_a-cool-post-xxxx") is None


def test_post_date_handles_none_and_empty_url():
    assert linkedin_post_date("") is None
    assert linkedin_post_date(None) is None


def test_post_date_compares_correctly_against_cutoff():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
    stale = linkedin_post_date(
        "https://www.linkedin.com/posts/x_y-activity-7397959444342575104-z")
    assert stale < cutoff


def test_handle_matches_first_token():
    """ASAL Technologies → handle 'asaltech' should match (first 3+ chars in 'asal')."""
    assert linkedin_handle_matches(
        "https://www.linkedin.com/posts/asaltech_hiring-something",
        "ASAL Technologies",
    ) is True


def test_handle_matches_iconnect():
    assert linkedin_handle_matches(
        "https://www.linkedin.com/posts/iconnect-tech_hiring-fullstack",
        "IConnect Technologies",
    ) is True


def test_handle_rejects_unrelated_post():
    """Generic 'Future' must not match 'pankh-workforce-solution'."""
    assert linkedin_handle_matches(
        "https://www.linkedin.com/posts/pankh-workforce-solution_internshipdrive",
        "Future Information Systems",
    ) is False


def test_handle_rejects_non_linkedin_url():
    assert linkedin_handle_matches(
        "https://example.com/jobs/123",
        "Anything",
    ) is False


def test_handle_handles_short_company_names():
    """A 1-2 char company name should never match (we require >=3-char token)."""
    assert linkedin_handle_matches(
        "https://www.linkedin.com/posts/ab_post",
        "AB",
    ) is False
