"""Tests for DEFAULT_AI_RESULT shape, _normalize_result, and _parse_ai_response.

The verdict schema is the contract between core_ai and every consumer (renderer,
DB, dashboard). Any silent change here would propagate to emails, GitHub Issues,
and future Stream-B web UI rows. Tests here gatekeep that contract.
"""
from core_ai import DEFAULT_AI_RESULT, _normalize_result, _parse_ai_response


def test_default_schema_keys():
    """The schema is exactly these 10 keys. Adding without updating consumers will break things."""
    expected = {
        "is_valid", "verdict", "tech_fit", "experience_fit", "logistics_fit",
        "match_percentage", "compensation", "effort", "suspicious", "scam",
    }
    assert set(DEFAULT_AI_RESULT.keys()) == expected


def test_normalize_full_dict():
    raw = {
        "is_valid": True, "verdict": "Good fit",
        "tech_fit": 90, "experience_fit": 70, "logistics_fit": 85,
        "match_percentage": 82, "compensation": "$25/hr",
        "effort": "low", "suspicious": False, "scam": False,
    }
    r = _normalize_result(raw)
    assert r["verdict"] == "Good fit"
    assert r["is_valid"] is True
    assert r["tech_fit"] == 90
    assert r["experience_fit"] == 70
    assert r["logistics_fit"] == 85
    assert r["match_percentage"] == 82
    assert r["compensation"] == "$25/hr"
    assert r["effort"] == "low"
    assert r["suspicious"] is False
    assert r["scam"] is False


def test_normalize_missing_fields_use_defaults():
    r = _normalize_result({})
    assert r["is_valid"] is False
    assert r["verdict"] == "No verdict"
    assert r["tech_fit"] == 0
    assert r["experience_fit"] == 0
    assert r["logistics_fit"] == 0
    assert r["match_percentage"] == 0
    assert r["compensation"] == "Not stated"
    assert r["effort"] == "unknown"
    assert r["suspicious"] is False
    assert r["scam"] is False


def test_normalize_non_dict_input():
    """Strings, None, lists — anything non-dict — fall back to all defaults."""
    assert _normalize_result("nope")["match_percentage"] == 0
    assert _normalize_result(None)["match_percentage"] == 0
    assert _normalize_result(["array"])["is_valid"] is False


def test_normalize_string_numbers_are_coerced():
    raw = {
        "tech_fit": "92%", "experience_fit": "65", "logistics_fit": "90.0",
        "match_percentage": "75.4%", "suspicious": "true", "is_valid": "yes",
    }
    r = _normalize_result(raw)
    assert r["tech_fit"] == 92
    assert r["experience_fit"] == 65
    assert r["logistics_fit"] == 90
    assert r["match_percentage"] == 75
    assert r["suspicious"] is True
    assert r["is_valid"] is True


def test_normalize_drops_unknown_keys():
    """Extra fields the AI invented should NOT leak into our schema."""
    raw = {"match_percentage": 80, "extra_field": "should be dropped", "score": 999}
    r = _normalize_result(raw)
    assert "extra_field" not in r
    assert "score" not in r
    assert r["match_percentage"] == 80


def test_parse_clean_json():
    s = (
        '{"is_valid": true, "verdict": "Strong RAG match", '
        '"tech_fit": 95, "experience_fit": 80, "logistics_fit": 90, '
        '"match_percentage": 90, "compensation": "$30/hr", '
        '"effort": "low", "suspicious": false, "scam": false}'
    )
    r = _parse_ai_response(s)
    assert r["match_percentage"] == 90
    assert r["verdict"] == "Strong RAG match"


def test_parse_with_json_markdown_fence():
    s = (
        '```json\n{"is_valid": true, "verdict": "ok", '
        '"tech_fit": 50, "experience_fit": 50, "logistics_fit": 50, '
        '"match_percentage": 50, "compensation": "Unpaid", '
        '"effort": "medium", "suspicious": false}\n```'
    )
    r = _parse_ai_response(s)
    assert r["compensation"] == "Unpaid"
    assert r["effort"] == "medium"


def test_parse_with_loose_fence():
    s = (
        '```\n{"is_valid": false, "verdict": "nope", '
        '"tech_fit": 10, "experience_fit": 10, "logistics_fit": 10, '
        '"match_percentage": 10, "compensation": "Not stated", '
        '"effort": "low", "suspicious": true}\n```'
    )
    r = _parse_ai_response(s)
    assert r["suspicious"] is True
    assert r["is_valid"] is False


def test_parse_empty_raises():
    raised = False
    try:
        _parse_ai_response("")
    except ValueError:
        raised = True
    assert raised
