"""multi_user_runner coercion helpers (B7).

The runner persists pandas-derived values straight into Supabase, which has
strict CHECK constraints (match_percentage 0..100, similarity 0..1, effort
in an enum, etc). The coercion helpers exist so a stray N/A / NaN / out-of-
range / odd-type value from the AI loop never trips a CHECK and kills the
whole INSERT batch. Locking the contract:
  * None in, None out
  * Out-of-range ints clamp to the schema bounds (don't drop the row)
  * effort drops anything outside the schema enum (NULL is legal)
  * similarity rounds + clamps to 0.0–1.0 (matches the numeric(5,4) column)
"""

import multi_user_runner as mur


def test_coerce_int_handles_none_and_garbage():
    assert mur._coerce_int(None) is None
    assert mur._coerce_int("N/A") is None
    assert mur._coerce_int("not a number") is None


def test_coerce_int_clamps_to_schema_range():
    assert mur._coerce_int(85) == 85
    assert mur._coerce_int(-5) == 0
    assert mur._coerce_int(200) == 100
    # Float input is common when pandas widens an int column with NaNs.
    assert mur._coerce_int(72.0) == 72


def test_coerce_bool_returns_none_for_none():
    assert mur._coerce_bool(None) is None
    assert mur._coerce_bool(True) is True
    assert mur._coerce_bool(False) is False


def test_coerce_similarity_clamps_and_rounds():
    assert mur._coerce_similarity(None) is None
    assert mur._coerce_similarity("not-a-float") is None
    assert mur._coerce_similarity(0.123456) == 0.1235
    assert mur._coerce_similarity(-0.1) == 0.0
    assert mur._coerce_similarity(1.5) == 1.0


def test_safe_effort_enforces_enum():
    assert mur._safe_effort("low") == "low"
    assert mur._safe_effort("Medium") == "medium"   # case-insensitive
    assert mur._safe_effort("HIGH") == "high"
    assert mur._safe_effort("unknown") == "unknown"
    assert mur._safe_effort("excruciating") is None  # not in enum → NULL
    assert mur._safe_effort(None) is None
    assert mur._safe_effort("") is None


def test_safe_str_strips_and_nones_empty():
    assert mur._safe_str(None) is None
    assert mur._safe_str("") is None
    assert mur._safe_str("   ") is None
    assert mur._safe_str("  Hello  ") == "Hello"
    assert mur._safe_str(42) == "42"


def test_jobs_to_rows_marks_ai_evaluated_flag():
    import pandas as pd
    df = pd.DataFrame([{
        "title": "Eng", "company": "Acme", "location": "Remote",
        "job_url": "https://x/1", "description": "x" * 5000,
        "ai_verdict": "Strong match", "is_valid": True,
        "match_percentage": 88, "tech_fit": 90, "experience_fit": 70,
        "logistics_fit": 85, "compensation": "$100k", "effort": "medium",
        "suspicious": False, "pre_flagged_low_quality": False,
        "pre_flagged_trusted": True, "similarity": 0.91234,
    }])

    rows = mur._jobs_to_rows(df, run_id=42, user_id="u1", ai_evaluated=True)
    assert len(rows) == 1
    r = rows[0]
    assert r["run_id"] == 42 and r["user_id"] == "u1"
    assert r["ai_evaluated"] is True
    assert r["ai_verdict"] == "Strong match"
    assert r["is_valid"] is True
    assert r["match_percentage"] == 88
    assert r["effort"] == "medium"
    assert r["similarity"] == 0.9123
    # description_excerpt is truncated to DESCRIPTION_EXCERPT_CHARS.
    assert len(r["description_excerpt"]) == mur.DESCRIPTION_EXCERPT_CHARS


def test_jobs_to_rows_lower_ranked_drops_ai_columns():
    import pandas as pd
    df = pd.DataFrame([{
        "title": "Eng", "company": "Acme", "location": "Remote",
        "job_url": "https://x/2", "description": "short",
        "ai_verdict": "should not propagate", "is_valid": True,
    }])
    rows = mur._jobs_to_rows(df, run_id=1, user_id="u1", ai_evaluated=False)
    assert rows[0]["ai_evaluated"] is False
    assert rows[0]["ai_verdict"] is None
    assert rows[0]["is_valid"] is None


def test_jobs_to_rows_handles_empty_df():
    import pandas as pd
    assert mur._jobs_to_rows(pd.DataFrame(), run_id=1, user_id="u1", ai_evaluated=True) == []
    assert mur._jobs_to_rows(None, run_id=1, user_id="u1", ai_evaluated=False) == []
