"""apply_post_ai_caps — deterministic match-percentage caps applied AFTER the
AI returns its verdict, BEFORE the optional network scam check.

Two caps in priority order:
  1. Reputation cap: pre_flagged_low_quality=True -> cap at 55, [BLACKLISTED] prefix
  2. AI-suspicious self-cap: suspicious=True + score > 55 -> cap at 55, [AI-SUSPICIOUS] prefix

These caps protect the email from sneaky-but-tech-keyword-rich postings that
the AI itself flags as suspicious but still gives high marks to. The fix
shipped 2026-05-17 after 4 India-sus internships (Inficore Soft, Skillzenloop)
leaked through into the daily email at match=80% / 60% / 60% / 59%.
"""
from pipeline.core_ai import apply_post_ai_caps


def _ai_result(match=85, suspicious=False, verdict="Looks fine"):
    return {
        "is_valid": True,
        "verdict": verdict,
        "tech_fit": match,
        "experience_fit": match,
        "logistics_fit": match,
        "match_percentage": match,
        "compensation": "Not stated",
        "effort": "low",
        "suspicious": suspicious,
        "scam": False,
    }


# ---------------------------------------------------------------------------
# Reputation cap (priority 1)
# ---------------------------------------------------------------------------

def test_blacklisted_company_capped_at_55():
    """The AI says 80% but the company is on the reputation blacklist."""
    result = _ai_result(match=80, verdict="Strong Python match.")
    row = {"pre_flagged_low_quality": True}
    out = apply_post_ai_caps(result, row)
    assert out["match_percentage"] == 55
    assert out["verdict"].startswith("[BLACKLISTED]")


def test_blacklisted_company_under_55_keeps_score():
    """Already below the cap — score should not be modified, only the prefix."""
    result = _ai_result(match=40, verdict="Mediocre fit.")
    row = {"pre_flagged_low_quality": True}
    out = apply_post_ai_caps(result, row)
    assert out["match_percentage"] == 40
    assert out["verdict"].startswith("[BLACKLISTED]")


def test_blacklisted_verdict_not_double_prefixed():
    """Re-running the cap should not stack [BLACKLISTED] prefixes."""
    result = _ai_result(match=80, verdict="[BLACKLISTED] something already")
    row = {"pre_flagged_low_quality": True}
    out = apply_post_ai_caps(result, row)
    assert out["verdict"].count("[BLACKLISTED]") == 1


def test_blacklisted_takes_precedence_over_ai_suspicious():
    """Even if the AI also marked suspicious=True, the blacklist tag is what shows."""
    result = _ai_result(match=80, suspicious=True, verdict="Looks sketchy.")
    row = {"pre_flagged_low_quality": True}
    out = apply_post_ai_caps(result, row)
    assert out["verdict"].startswith("[BLACKLISTED]")
    assert "[AI-SUSPICIOUS]" not in out["verdict"]


# ---------------------------------------------------------------------------
# AI-suspicious self-cap (priority 2) — the new Fix #5
# ---------------------------------------------------------------------------

def test_ai_suspicious_high_score_capped_at_55():
    """Inficore Soft pattern from 2026-05-17: AI gave 80% AND marked suspicious."""
    result = _ai_result(match=80, suspicious=True, verdict="Strong Python match but low-value internship.")
    row = {}  # not on the reputation blacklist
    out = apply_post_ai_caps(result, row)
    assert out["match_percentage"] == 55
    assert out["verdict"].startswith("[AI-SUSPICIOUS]")


def test_ai_suspicious_score_at_60_capped():
    """Score equal to 60 (just over the 55 cap) should still be lowered."""
    result = _ai_result(match=60, suspicious=True)
    out = apply_post_ai_caps(result, {})
    assert out["match_percentage"] == 55


def test_ai_suspicious_score_at_55_left_alone():
    """Already at-or-below the cap — score stays, but the verdict gets tagged."""
    result = _ai_result(match=55, suspicious=True, verdict="Decent borderline match.")
    out = apply_post_ai_caps(result, {})
    assert out["match_percentage"] == 55
    # The verdict gets tagged only if score WAS > 55 — at 55, do not re-tag.
    # (Saves cosmetic noise; the AI-suspicious flag is already on the row.)
    assert not out["verdict"].startswith("[AI-SUSPICIOUS]")


def test_ai_suspicious_low_score_left_alone():
    """A 40% suspicious result is already below the cap; nothing to clamp."""
    result = _ai_result(match=40, suspicious=True, verdict="Doesn't really fit.")
    out = apply_post_ai_caps(result, {})
    assert out["match_percentage"] == 40
    assert not out["verdict"].startswith("[AI-SUSPICIOUS]")


def test_not_suspicious_and_not_blacklisted_unchanged():
    """A clean high-score result must not be touched."""
    result = _ai_result(match=92, suspicious=False, verdict="Excellent match.")
    out = apply_post_ai_caps(result, {})
    assert out["match_percentage"] == 92
    assert out["verdict"] == "Excellent match."


def test_ai_suspicious_verdict_not_double_prefixed():
    """Idempotency: re-running the cap should not stack [AI-SUSPICIOUS]."""
    result = _ai_result(match=80, suspicious=True, verdict="[AI-SUSPICIOUS] already-tagged")
    out = apply_post_ai_caps(result, {})
    assert out["verdict"].count("[AI-SUSPICIOUS]") == 1


def test_ai_suspicious_does_not_prefix_if_scam_tag_already_present():
    """If the verdict was somehow already [SCAM]-tagged, don't add a weaker tag in front."""
    result = _ai_result(match=80, suspicious=True, verdict="[SCAM] confirmed via web search")
    out = apply_post_ai_caps(result, {})
    # Cap still applies — but no double-prefix
    assert out["match_percentage"] == 55
    assert out["verdict"].startswith("[SCAM]")
    assert "[AI-SUSPICIOUS]" not in out["verdict"]


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------

def test_non_dict_result_passes_through():
    """Garbage input doesn't crash — returns the input unchanged."""
    assert apply_post_ai_caps(None, {}) is None
    assert apply_post_ai_caps("not a dict", {}) == "not a dict"
