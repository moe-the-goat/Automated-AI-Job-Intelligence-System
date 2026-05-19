"""Generalized geo-lock disqualifier patterns (added 2026-05-19).

quick_viability_check() now drops jobs whose description contains an explicit
country/region exclusion (Canadian-only, German-only, EU-only, etc.) without
spending an LLM call. These tests lock down the regex behavior so a tweak to
the country list or phrase templates can't regress real cases.

The false-positive guards at the bottom are the most important tests — a
geo-restriction regex that fires on `our team is distributed across Europe`
would delete most of our top matches.
"""
from pipeline.core_ai import quick_viability_check


# Padding helps each description clear the _MIN_DESCRIPTION_CHARS=150 floor so
# the test isn't accidentally rejected by step 3 instead of the geo step.
_PAD = " padding text here for length. " * 8


# ---------------------------------------------------------------------------
# Anglosphere locks
# ---------------------------------------------------------------------------

def test_rejects_canadian_residents_only():
    row = {"title": "AI Engineer", "description": _PAD + "Canadian residents only."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "geo-locked" in reason


def test_rejects_must_be_based_in_australia():
    row = {"title": "ML Engineer", "description": _PAD + "Candidates must be based in Australia."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "geo-locked" in reason


def test_rejects_uk_only_role():
    row = {"title": "Backend Engineer", "description": _PAD + "This is a UK-only role."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "geo-locked" in reason


def test_rejects_must_reside_in_germany():
    row = {"title": "AI Engineer", "description": _PAD + "Must reside in Germany for tax purposes."}
    ok, reason = quick_viability_check(row)
    assert ok is False


# ---------------------------------------------------------------------------
# Regional locks (EU, LATAM, APAC)
# ---------------------------------------------------------------------------

def test_rejects_eu_residents_only():
    row = {"title": "ML Engineer", "description": _PAD + "EU residents only please."}
    ok, reason = quick_viability_check(row)
    assert ok is False
    assert "geo-locked" in reason


def test_rejects_only_open_to_candidates_in_latam():
    row = {"title": "Software Engineer",
           "description": _PAD + "This role is only available to candidates in LATAM."}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_apac_candidates_only():
    row = {"title": "Data Scientist",
           "description": _PAD + "We are hiring APAC candidates only for this role."}
    ok, _ = quick_viability_check(row)
    assert ok is False


# ---------------------------------------------------------------------------
# Latin America / Asia
# ---------------------------------------------------------------------------

def test_rejects_brazil_based_candidates_only():
    row = {"title": "Junior Software Engineer",
           "description": _PAD + "Brazil-based candidates only for this remote role."}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_must_be_located_in_mexico():
    row = {"title": "Software Engineer",
           "description": _PAD + "You must be located in Mexico to be eligible."}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_india_residents_only():
    row = {"title": "Data Analyst",
           "description": _PAD + "India residents only — local payroll required."}
    ok, _ = quick_viability_check(row)
    assert ok is False


def test_rejects_must_have_right_to_work_in_australia():
    row = {"title": "Engineer",
           "description": _PAD + "Must have the right to work in Australia without sponsorship."}
    ok, _ = quick_viability_check(row)
    assert ok is False


# ---------------------------------------------------------------------------
# Israel (specifically — Palestine candidates categorically excluded)
# ---------------------------------------------------------------------------

def test_rejects_israeli_citizens_only():
    row = {"title": "Backend Engineer",
           "description": _PAD + "Israeli citizens only for this defense-adjacent role."}
    ok, _ = quick_viability_check(row)
    assert ok is False


# ---------------------------------------------------------------------------
# CRITICAL: false-positive guards
# ---------------------------------------------------------------------------
# A geo-restriction regex that fires on routine mentions of countries (HQ in
# London, distributed across Europe, etc.) would delete real matches. These
# tests lock down the behavior — they MUST keep passing.

def test_keeps_globally_distributed_team_mentioning_germany():
    """`our team is distributed across Germany and the UK` is not a restriction."""
    row = {"title": "ML Engineer",
           "description": _PAD + "Our globally remote team includes engineers in Germany, UK, "
                                 "and the Americas. Welcome candidates worldwide."}
    ok, reason = quick_viability_check(row)
    assert ok is True, f"False positive: rejected with reason {reason!r}"


def test_keeps_hq_in_canada_mention():
    """`HQ in Toronto, Canada` alone doesn't mean Canadian-only."""
    row = {"title": "Software Engineer",
           "description": _PAD + "Our HQ is in Canada but we hire fully remote globally. "
                                 "Welcome candidates from anywhere in the world."}
    ok, reason = quick_viability_check(row)
    assert ok is True, f"False positive: rejected with reason {reason!r}"


def test_keeps_team_in_brazil_and_argentina_mention():
    """Multiple country names in a 'team locations' context shouldn't disqualify."""
    row = {"title": "Backend Engineer",
           "description": _PAD + "We have team members in Brazil, Argentina, Mexico and Spain "
                                 "and welcome remote workers from any location."}
    ok, reason = quick_viability_check(row)
    assert ok is True, f"False positive: rejected with reason {reason!r}"


def test_keeps_mena_friendly_role():
    """Middle East / MENA is NOT in the disqualifier list — Palestine is in MENA."""
    row = {"title": "AI Engineer",
           "description": _PAD + "We welcome remote workers from the Middle East / MENA region "
                                 "as part of our globally distributed engineering team."}
    ok, _ = quick_viability_check(row)
    assert ok is True


def test_keeps_palestine_friendly_explicit():
    """Explicit Palestine mention should never be disqualifying."""
    row = {"title": "Software Engineer",
           "description": _PAD + "We hire fully remote from anywhere including Palestine, "
                                 "Jordan, Egypt and the broader MENA region."}
    ok, _ = quick_viability_check(row)
    assert ok is True
