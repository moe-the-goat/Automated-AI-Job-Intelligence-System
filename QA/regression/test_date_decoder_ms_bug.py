"""REGRESSION: LinkedIn activity-ID date decoder ms-vs-seconds bug (2026-05-16).

Symptom user saw: 5-month-old and 1-year-old LinkedIn posts arrived in the
local-companies email even though LOCAL_LOOKBACK_DAYS = 6.

Root cause: linkedin_post_date treated `activity_id >> 22` as Unix epoch
seconds. It's actually Unix epoch MILLISECONDS — the top 41 bits of the
snowflake ID. `datetime.fromtimestamp(1.76e12)` raised OverflowError, the
function's except returned None, and the filter `if post_date and post_date < cutoff`
never tripped.

Fix: divide by 1000 before fromtimestamp.

If this test ever fails, someone has reintroduced the bug. Don't "fix" it
by removing the test — trace why the dates aren't decoding correctly.
"""
from datetime import datetime, timezone, timedelta
from pipeline.core_local_sources import linkedin_post_date, LOCAL_LOOKBACK_DAYS


# Two real activity URLs from the email the user flagged on 2026-05-16:
ICONNECT_URL = "https://www.linkedin.com/posts/iconnect-tech_hiring-fullstackdeveloper-ai-activity-7397959444342575104-p6DK"
SMARTWEB_URL = "https://www.linkedin.com/posts/smartweb-labs_hiring-python-python-activity-7284632728031944704-93L4"


def test_iconnect_post_decodes_to_november_2025_not_none():
    """The pre-fix code returned None here, so the filter passed the stale post through."""
    d = linkedin_post_date(ICONNECT_URL)
    assert d is not None, "Date decoder returned None — the ms-vs-s bug is back."
    assert d.year == 2025
    assert d.month == 11   # Nov 22, 2025 — user said '5 months old'


def test_smartweb_post_decodes_to_january_2025():
    """The pre-fix code returned None here, so the filter passed the stale post through."""
    d = linkedin_post_date(SMARTWEB_URL)
    assert d is not None
    assert d.year == 2025
    assert d.month == 1    # Jan 13, 2025 — user said '1 year old'


def test_iconnect_post_is_before_cutoff():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
    assert linkedin_post_date(ICONNECT_URL) < cutoff


def test_smartweb_post_is_before_cutoff():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
    assert linkedin_post_date(SMARTWEB_URL) < cutoff


def test_decoded_date_is_within_reasonable_year_bounds():
    """Sanity guard: the decoded year must be between 2010 (LinkedIn's launch era) and
    2050. If it's ~57000, someone forgot the /1000 again."""
    d = linkedin_post_date(ICONNECT_URL)
    assert 2010 <= d.year <= 2050, f"Decoded year {d.year} is implausible — ms-vs-s bug?"
