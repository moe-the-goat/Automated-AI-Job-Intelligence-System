"""local_companies.is_palestine_or_remote — the Option-B area-sweep geo guard.

JobSpy, on a low-supply location like Palestine, falls back to returning global
jobs (a flood of US/Indiana roles slipped into a real email). This guard keeps
only Palestine-based or genuinely-remote jobs from the broad area sweep. Locking
it so the flood can't return.
"""
from local_companies import is_palestine_or_remote


def test_keeps_palestine_cities():
    for loc in ["Ramallah, Palestine", "Nablus", "Gaza Strip", "Hebron",
                "Birzeit", "West Bank", "Bethlehem"]:
        assert is_palestine_or_remote(loc) is True, loc


def test_keeps_remote_and_worldwide():
    assert is_palestine_or_remote("Remote") is True
    assert is_palestine_or_remote("Anywhere") is True
    assert is_palestine_or_remote("Worldwide") is True


def test_keeps_remote_in_title_even_if_location_foreign_ish():
    # A remote role whose location field is vague but title says remote.
    assert is_palestine_or_remote("", title="Remote Software Engineer") is True


def test_keeps_unknown_location():
    # Unknown/empty -> let the AI decide (don't silently drop local-board jobs).
    assert is_palestine_or_remote("") is True
    assert is_palestine_or_remote(None) is True
    assert is_palestine_or_remote("nan") is True


def test_drops_explicit_foreign_locations():
    # These are exactly the jobs that flooded the email.
    for loc in ["Indianapolis, IN", "Indiana, United States", "San Francisco, CA",
                "London, United Kingdom", "Berlin, Germany", "Bangalore, India"]:
        assert is_palestine_or_remote(loc) is False, loc


def test_foreign_location_not_rescued_by_unrelated_title():
    assert is_palestine_or_remote("Indianapolis, IN", title="Backend Engineer") is False
