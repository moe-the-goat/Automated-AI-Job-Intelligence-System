"""core_geo: Layer 3 Gemini geo-eligibility check.

The actual Gemini call is mocked out — we test the trigger function, the
response parser, and the result-application logic without touching the network.
"""
from pipeline.core_geo import (
    should_geo_check,
    _parse_geo_response,
    apply_geo_check_result,
)


# ---------------------------------------------------------------------------
# should_geo_check — the trigger function
# ---------------------------------------------------------------------------

def test_triggers_on_remote_country_in_title():
    row = {"title": "Junior Software Engineer — Remote, Mexico",
           "location": "",
           "description": "We are hiring..."}
    assert should_geo_check(row) is True


def test_triggers_on_remote_brazil_in_title():
    row = {"title": "Junior Engineer - Remote, Brazil",
           "location": "",
           "description": "We hire remote workers globally..."}
    assert should_geo_check(row) is True


def test_triggers_on_parenthesized_eu_in_title():
    row = {"title": "Junior Web Developer (EU)",
           "location": "",
           "description": "Build web apps..."}
    assert should_geo_check(row) is True


def test_triggers_on_specific_location_field():
    row = {"title": "Software Engineer",
           "location": "London, UK",
           "description": "Build apps... worldwide team..."}
    assert should_geo_check(row) is True


def test_triggers_when_no_global_confirmer_in_description():
    """Plain title and empty location, but no 'worldwide' phrase — still check."""
    row = {"title": "Software Engineer",
           "location": "",
           "description": "Build amazing systems. Great team. Apply now."}
    assert should_geo_check(row) is True


def test_no_trigger_when_explicitly_worldwide():
    row = {"title": "Software Engineer",
           "location": "Remote",
           "description": "We hire remote workers worldwide with no geographic restrictions."}
    assert should_geo_check(row) is False


def test_no_trigger_when_emea_welcome():
    row = {"title": "ML Engineer",
           "location": "Remote",
           "description": "EMEA welcome - we hire across Middle East, Europe and Africa."}
    assert should_geo_check(row) is False


def test_no_trigger_when_mena_explicit():
    row = {"title": "Software Engineer",
           "location": "Worldwide",
           "description": "Remote-first team welcoming candidates from MENA and beyond."}
    assert should_geo_check(row) is False


# ---------------------------------------------------------------------------
# _parse_geo_response — response parsing tolerance
# ---------------------------------------------------------------------------

def test_parses_clean_json():
    text = '{"eligibility": "open", "confidence": 85, "evidence": "Globally remote per LinkedIn."}'
    result = _parse_geo_response(text)
    assert result["eligibility"] == "open"
    assert result["confidence"] == 85
    assert "LinkedIn" in result["evidence"]


def test_parses_json_with_markdown_fence():
    text = '```json\n{"eligibility": "restricted", "confidence": 90, "evidence": "Mexico-only."}\n```'
    result = _parse_geo_response(text)
    assert result["eligibility"] == "restricted"
    assert result["confidence"] == 90


def test_parses_json_with_bare_code_fence():
    text = '```\n{"eligibility": "uncertain", "confidence": 50, "evidence": "Insufficient info."}\n```'
    result = _parse_geo_response(text)
    assert result["eligibility"] == "uncertain"
    assert result["confidence"] == 50


def test_extracts_json_from_surrounding_prose():
    """Sometimes Gemini writes a sentence before the JSON even when told not to."""
    text = ('Based on my search, here is the answer: '
            '{"eligibility": "open", "confidence": 75, "evidence": "Found global remote policy."}')
    result = _parse_geo_response(text)
    assert result["eligibility"] == "open"
    assert result["confidence"] == 75


def test_falls_back_to_uncertain_on_garbage():
    """Malformed responses MUST default to uncertain — never silently 'open'."""
    result = _parse_geo_response("absolute nonsense response with no json at all")
    assert result["eligibility"] == "uncertain"
    assert result["confidence"] == 0


def test_falls_back_to_uncertain_on_empty():
    result = _parse_geo_response("")
    assert result["eligibility"] == "uncertain"
    assert result["confidence"] == 0


def test_normalizes_unknown_eligibility_value():
    """If Gemini hallucinates a status outside the allowed enum, snap to uncertain."""
    text = '{"eligibility": "maybe", "confidence": 60, "evidence": "ambiguous"}'
    result = _parse_geo_response(text)
    assert result["eligibility"] == "uncertain"


def test_clamps_confidence_to_0_100():
    text = '{"eligibility": "open", "confidence": 999, "evidence": "test"}'
    result = _parse_geo_response(text)
    assert result["confidence"] == 100

    text2 = '{"eligibility": "open", "confidence": -50, "evidence": "test"}'
    result2 = _parse_geo_response(text2)
    assert result2["confidence"] == 0


def test_coerces_string_confidence():
    """Some model outputs put confidence as `"85"` or `"85%"`."""
    text = '{"eligibility": "open", "confidence": "85%", "evidence": "test"}'
    result = _parse_geo_response(text)
    assert result["confidence"] == 85


def test_truncates_long_evidence():
    long_text = "a" * 500
    result = _parse_geo_response(
        f'{{"eligibility": "open", "confidence": 70, "evidence": "{long_text}"}}'
    )
    assert len(result["evidence"]) <= 300


# ---------------------------------------------------------------------------
# apply_geo_check_result — merging the geo verdict into the main result
# ---------------------------------------------------------------------------

def _base_verdict(match=80, logistics=90):
    """A canonical AI-validated verdict used as the input to apply_geo_check_result."""
    return {
        "is_valid": True,
        "verdict": "MATCH: RAG project aligns. GAP: no AWS experience.",
        "tech_fit": 85,
        "experience_fit": 75,
        "logistics_fit": logistics,
        "match_percentage": match,
        "compensation": "$50/hr",
        "effort": "low",
        "suspicious": False,
        "scam": False,
    }


def test_open_eligibility_makes_no_change():
    verdict = _base_verdict(match=85, logistics=90)
    geo = {"eligibility": "open", "confidence": 80, "evidence": "Globally remote."}
    result = apply_geo_check_result(verdict, geo)
    assert result["is_valid"] is True
    assert result["match_percentage"] == 85
    assert result["logistics_fit"] == 90
    assert "[GEO-" not in result["verdict"]


def test_uncertain_caps_match_at_50():
    verdict = _base_verdict(match=85)
    geo = {"eligibility": "uncertain", "confidence": 40, "evidence": "No public info."}
    result = apply_geo_check_result(verdict, geo)
    assert result["match_percentage"] == 50, "match_percentage must cap at 50 for uncertain"
    assert result["is_valid"] is True, "uncertain shouldn't drop the job — only cap the score"
    assert result["verdict"].startswith("[GEO-UNVERIFIED]")
    assert "No public info." in result["verdict"]


def test_uncertain_below_50_does_not_inflate():
    """Uncertain should CAP at 50, not raise low scores to 50."""
    verdict = _base_verdict(match=35)
    geo = {"eligibility": "uncertain", "confidence": 0, "evidence": ""}
    result = apply_geo_check_result(verdict, geo)
    assert result["match_percentage"] == 35


def test_restricted_drops_job_with_full_cap():
    verdict = _base_verdict(match=85, logistics=90)
    geo = {"eligibility": "restricted", "confidence": 95, "evidence": "Mexico residents only per LinkedIn."}
    result = apply_geo_check_result(verdict, geo)
    assert result["is_valid"] is False, "restricted MUST set is_valid=False so email filter drops it"
    assert result["logistics_fit"] <= 15
    assert result["match_percentage"] <= 30
    assert result["verdict"].startswith("[GEO-RESTRICTED]")
    assert "Mexico" in result["verdict"]


def test_scam_tag_wins_over_geo_tag():
    """An existing [SCAM] tag must not be overwritten by the geo cap."""
    verdict = _base_verdict(match=30)
    verdict["verdict"] = "[SCAM] India-job-mill posting flagged"
    geo = {"eligibility": "restricted", "confidence": 95, "evidence": "India-only."}
    result = apply_geo_check_result(verdict, geo)
    # [GEO-RESTRICTED] should NOT have been prepended on top of [SCAM]
    assert not result["verdict"].startswith("[GEO-RESTRICTED]")
    assert result["verdict"].startswith("[SCAM]")
    # But the numeric caps still apply
    assert result["is_valid"] is False


def test_blacklisted_tag_wins_over_geo_tag():
    verdict = _base_verdict(match=55)
    verdict["verdict"] = "[BLACKLISTED] WebBoost Solutions IT"
    geo = {"eligibility": "uncertain", "confidence": 30, "evidence": "no info found"}
    result = apply_geo_check_result(verdict, geo)
    assert not result["verdict"].startswith("[GEO-UNVERIFIED]")
    assert result["verdict"].startswith("[BLACKLISTED]")


def test_handles_non_dict_inputs_gracefully():
    """Bad inputs shouldn't crash — just no-op."""
    assert apply_geo_check_result(None, {"eligibility": "open"}) is None
    assert apply_geo_check_result({"is_valid": True}, None) == {"is_valid": True}
