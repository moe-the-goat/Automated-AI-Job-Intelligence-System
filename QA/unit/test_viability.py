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


def test_keeps_ddg_snippet_with_short_body():
    """DDG-sourced local posts legitimately have hashtag-teaser bodies (tagged
    source='ddg_*'). Don't reject those for being short."""
    row = {"title": "We're hiring a React developer",
           "source": "ddg_linkedin",
           "description": "#hiring #python #react #developer #remote"}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_legacy_linkedin_post_prefix():
    """Backward-compat: the old 'LinkedIn Post:' title prefix still bypasses the
    short-body check, in case any cached/old row carries it."""
    row = {"title": "LinkedIn Post: #hiring #python #react ...",
           "description": "#hiring #python #react #developer #remote"}
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


# ---------------------------------------------------------------------------
# Hard work-auth / clearance disqualifiers (Fix #3b)
# ---------------------------------------------------------------------------
# Palestine-based candidate cannot legally accept US-only / US-clearance roles,
# so we drop them at the pre-screen instead of wasting a Gemini call.

def test_rejects_us_citizens_only():
    row = {"title": "AI Intern",
           "description": "Great team " + ("padding text. " * 10) + "US citizens only please."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "hard disqualifier" in reason
    assert "US citizens only" in reason


def test_rejects_must_be_us_citizen():
    row = {"title": "Software Engineer",
           "description": "Great role with lots of work " + ("padding. " * 15) + "Candidate must be a US citizen."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "US citizen" in reason


def test_rejects_must_reside_in_us():
    row = {"title": "AI Intern",
           "description": "Build awesome ML systems " + ("padding. " * 15) + "Must reside in the United States."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "reside in the US" in reason


def test_rejects_must_be_authorized_to_work_in_us():
    row = {"title": "Software Engineer",
           "description": "Awesome distributed team " + ("padding. " * 15) + "Must be authorized to work in the United States."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "authorized to work in the US" in reason


def test_rejects_no_visa_sponsorship():
    row = {"title": "Backend Developer",
           "description": "Great role for the right candidate. " + ("padding. " * 15) + "No visa sponsorship will be provided."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "visa sponsorship" in reason


def test_rejects_security_clearance_required():
    row = {"title": "Software Engineer",
           "description": "Build government systems with our team. " + ("padding. " * 15) + "Security clearance required."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "clearance" in reason


def test_rejects_ts_sci_clearance():
    row = {"title": "AI Engineer",
           "description": "Work on classified ML projects with us. " + ("padding. " * 15) + "Must have an active TS/SCI clearance."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "TS/SCI" in reason


def test_rejects_us_based_candidates_only_phrase():
    row = {"title": "Software Engineer",
           "description": "Distributed remote team building cool stuff. " + ("padding. " * 15) + "US-based candidates only."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "US-based candidates only" in reason


def test_keeps_generic_us_mention_without_disqualifier():
    """`headquartered in the US` alone should NOT disqualify — only specific phrases do.

    This is the critical false-positive guard. A globally remote company can mention
    that its HQ is in the US without that meaning the role is US-only.
    """
    row = {"title": "Software Engineer",
           "description": "Our team is fully distributed; HQ is in the United States but we hire globally and welcome candidates from EMEA, APAC, and the Americas. "
                          "Build great products with cutting-edge tech." + (" extra. " * 8)}
    ok, reason = quick_viability_check(row)
    assert ok is True, f"False positive: rejected with reason {reason!r}"


def test_keeps_globally_remote_with_us_headquarters():
    """Another flavor of the false-positive guard."""
    row = {"title": "ML Engineer",
           "description": "We are a US-headquartered company but the team works fully remote across the world. " + ("padding. " * 10)}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_hard_disqualifier_runs_after_senior_check():
    """A row that's both senior AND US-citizen-only should still be rejected.

    Senior check runs first (step 4) and short-circuits, so the senior reason
    is what we expect to see. Either reason is acceptable as long as the row
    is rejected — we just want to confirm both filters are wired in.
    """
    row = {"title": "Software Engineer",
           "description": "We require 10+ years of experience. " + ("padding text here. " * 20) + "US citizens only."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "senior" in reason or "hard disqualifier" in reason
