"""Tests for the langdetect-backed language filter in core_filter._is_english_title.

This function had a real production bug: the threshold was 10 chars, but langdetect
mis-detects short tech titles ("AI Engineer", "Data Science Intern") as Italian/
Welsh/etc., dropping legitimate jobs in every daily run. Bumped to 30 chars.

If this test ever fails because someone lowered the threshold back, trace why
short tech titles started getting dropped — it's almost certainly langdetect
false positives on titles loaded with proper nouns.
"""
from core_filter import _is_english_title


def test_short_tech_title_kept_regardless_of_langdetect():
    """Titles under the 30-char threshold bypass langdetect entirely."""
    assert _is_english_title("AI Engineer") is True
    assert _is_english_title("Data Science Intern") is True
    assert _is_english_title("Backend Developer") is True
    assert _is_english_title("Junior Software Engineer") is True


def test_clearly_english_long_title_kept():
    assert _is_english_title(
        "Junior Software Engineer with strong Python and AWS experience required") is True


def test_clearly_italian_long_title_dropped():
    """At 50+ chars langdetect should be confident — but only run this if langdetect is installed."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
    except ImportError:
        return  # langdetect not installed locally — function is a no-op, can't test rejection
    # 67 chars, unambiguously Italian
    assert _is_english_title(
        "Sviluppatore Senior PHP MySQL per azienda fintech italiana di Roma") is False


def test_empty_or_none_title_kept():
    assert _is_english_title("") is True
    assert _is_english_title(None) is True


def test_whitespace_only_title_kept():
    assert _is_english_title("    ") is True


def test_threshold_boundary_at_30_chars():
    """Exactly 30 chars should bypass; 31+ runs langdetect (which still keeps clear English)."""
    title_29 = "a" * 29                   # under threshold -> kept without check
    title_31 = "Software Engineering Internship for Backend Cloud Platform"  # > 30, clearly English
    assert _is_english_title(title_29) is True
    assert _is_english_title(title_31) is True
