"""REGRESSION: pre-screen used to reject every API-sourced job with missing description.

Symptom on 2026-05-16: global workflow skipped 33 of 35 jobs in the pre-screen
with reason `description too short (4 chars)`. The "4 chars" was the literal
string "nan" — pandas/API gave None and `str(None) == "None"`, `str(NaN) == "nan"`.

The AI's `get_full_job_description` URL-fetch fallback exists specifically to
handle this case, but the pre-screen never let the job reach the AI.

Fix: bypass the short-description rule when description is one of
{"", "nan", "none", "null"} OR has fewer than 20 chars, OR is a DDG-sourced
local snippet (source="ddg_*", or legacy "LinkedIn Post:" title prefix), OR is
the explicit "[NO DESCRIPTION]" / "[DESCRIPTION TRUNCATED]" placeholder.
"""
from pipeline.core_ai import quick_viability_check


def test_nan_description_passes_to_ai():
    """Pandas NaN stringified is 'nan'. Must not be flagged as 'too short'."""
    row = {"title": "AI Engineering Intern", "description": "nan"}
    ok, reason = quick_viability_check(row)
    assert ok is True


def test_None_string_description_passes():
    row = {"title": "AI Engineering Intern", "description": "None"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_null_string_description_passes():
    row = {"title": "AI Engineering Intern", "description": "null"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_empty_description_passes():
    row = {"title": "AI Engineering Intern", "description": ""}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_explicit_no_description_placeholder_passes():
    row = {"title": "AI Engineering Intern",
           "description": "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_truncated_description_placeholder_passes():
    row = {"title": "AI Engineering Intern",
           "description": "[DESCRIPTION TRUNCATED BY LINKEDIN LOGIN WALL]"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_ddg_sourced_snippet_with_hashtag_body_passes():
    """DDG-sourced local snippets (source='ddg_*') are legitimately short."""
    row = {"title": "We're hiring a backend developer",
           "source": "ddg_website",
           "description": "#hiring #python #react #developer #remote"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_legacy_linkedin_post_prefix_still_passes():
    """Backward-compat for any cached row still carrying the old title prefix."""
    row = {"title": "LinkedIn Post: #hiring #python #react ...",
           "description": "#hiring #python #react #developer #remote"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_substantive_short_description_still_rejected():
    """20-149 char descriptions that are NOT placeholder strings are still skipped."""
    row = {"title": "AI Intern",
           "description": "Apply now for this great role at our company."}  # 47 chars
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "chars" in reason
