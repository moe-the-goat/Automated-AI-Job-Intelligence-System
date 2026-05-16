"""Helpers backing the India + suspicious → scam detection in core_ai."""
from core_ai import looks_like_india_employer, scan_for_scam_signals


def test_india_employer_by_location_india():
    assert looks_like_india_employer("Bangalore, India", "Foo Corp") is True


def test_india_employer_by_indian_city():
    assert looks_like_india_employer("Bengaluru", "X") is True
    assert looks_like_india_employer("Mumbai", "X") is True
    assert looks_like_india_employer("Hyderabad", "X") is True
    assert looks_like_india_employer("Noida", "X") is True


def test_india_employer_by_company_suffix_private_limited():
    assert looks_like_india_employer(
        "Remote", "Zetheta Algorithms Private Limited") is True


def test_india_employer_by_company_suffix_pvt_ltd():
    assert looks_like_india_employer("Remote", "Acme Tech Pvt Ltd") is True
    assert looks_like_india_employer("Remote", "Beta Corp Pvt. Ltd") is True


def test_india_employer_us_not_flagged():
    assert looks_like_india_employer("San Francisco, CA", "Anthropic") is False


def test_india_employer_eu_not_flagged():
    assert looks_like_india_employer("Berlin, Germany", "Hugging Face GmbH") is False


def test_india_employer_empty_inputs():
    assert looks_like_india_employer("", "") is False


def test_scam_scan_finds_multiple_signals():
    text = (
        "I worked there for two months and it turned out to be a scam, they never paid me. "
        "Lots of reddit threads about this being a fake job."
    )
    assert scan_for_scam_signals(text) is True


def test_scam_scan_below_threshold():
    """One mention of 'scam' isn't enough — generic articles trip false positives otherwise."""
    text = "Top 10 ways to spot a job scam in 2026: tip 1, tip 2..."
    assert scan_for_scam_signals(text) is False


def test_scam_scan_empty_text():
    assert scan_for_scam_signals("") is False


def test_scam_scan_none_text():
    assert scan_for_scam_signals(None) is False


def test_scam_scan_custom_threshold():
    text = "Just a single mention of scam here."
    assert scan_for_scam_signals(text, min_matches=1) is True
    assert scan_for_scam_signals(text, min_matches=2) is False
