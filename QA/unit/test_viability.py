"""quick_viability_check — the cheap pre-screen that decides whether a job
deserves a full Gemini call.

Bugs here cost real money: too strict and we skip legitimate jobs without
any verdict; too lax and we burn Gemini quota on garbage.
"""
from pipeline.core_ai import quick_viability_check, skipped_result, DEFAULT_AI_RESULT


GOOD_DESCRIPTION = (
    "Build production RAG systems with LangChain and FastAPI. We deploy ML "
    "models on AWS and our team is fully distributed across EMEA timezones. "
    "Looking for candidates passionate about LLM applications."
)


def test_viable_when_normal_tech_job():
    ok, reason = quick_viability_check({"title": "AI Engineering Intern", "description": GOOD_DESCRIPTION})
    assert ok is True
    assert reason == "viable"


def test_short_circuits_on_blacklist_flag():
    """Reputation blacklist should bypass every other check."""
    row = {"title": "AI Engineering Intern", "description": GOOD_DESCRIPTION,
           "pre_flagged_low_quality": True}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "reputation" in reason


def test_rejects_sales_title():
    ok, reason = quick_viability_check({"title": "Sales Engineer", "description": GOOD_DESCRIPTION})
    assert ok is False
    assert "non-tech" in reason


def test_rejects_recruiter_title():
    row = {"title": "Technical Recruiter Intern", "description": GOOD_DESCRIPTION}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_marketing_title():
    row = {"title": "Marketing Engineer", "description": GOOD_DESCRIPTION}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_substantive_short_description():
    """A real but lazy-repost-style description in 20-149 chars range gets rejected."""
    row = {"title": "AI Intern",
           "description": "Apply now for this great role at our company."}  # 47 chars
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "chars" in reason


def test_keeps_nan_description():
    """API-sourced rows with description='nan' must pass — let AI URL-fetch fallback try."""
    row = {"title": "AI Engineering Intern", "description": "nan"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_none_string_description():
    row = {"title": "AI Engineering Intern", "description": "None"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_empty_description():
    row = {"title": "AI Engineering Intern", "description": ""}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_linkedin_post_with_short_body():
    """LinkedIn-radar posts legitimately have hashtag-teaser bodies. Don't reject those."""
    row = {"title": "LinkedIn Post: #hiring #python #react ...",
           "description": "#hiring #python #python #react #developer #remote"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_explicit_no_description_placeholder():
    """When core_ai already marked it [NO DESCRIPTION], let it through — Limited Info Protocol handles it."""
    row = {"title": "AI Engineering Intern",
           "description": "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_rejects_5_plus_years_experience():
    row = {"title": "AI Intern",
           "description": "We require 5+ years of experience in production ML. " * 5}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "senior experience" in reason


def test_rejects_minimum_7_years():
    row = {"title": "AI Intern",
           "description": "Minimum of 7 years experience required in this field. " * 5}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_principal_engineer_in_description():
    row = {"title": "AI Intern",
           "description": "You'll be a principal engineer leading the AI team. " * 5}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_skipped_result_matches_schema():
    """skipped_result must return the canonical schema so callers can blindly use it."""
    r = skipped_result("test reason")
    assert set(r.keys()) == set(DEFAULT_AI_RESULT.keys())
    assert r["is_valid"] is False
    assert r["verdict"].startswith("Pre-screen skipped:")
    assert r["match_percentage"] == 0
