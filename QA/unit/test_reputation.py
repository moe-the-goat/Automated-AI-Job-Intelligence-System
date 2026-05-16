"""Reputation prefilter: catches known job-mill companies before they reach AI."""
import pandas as pd
from core_filter import _pre_flag_reputation, _load_reputation


def test_load_returns_expected_shape():
    rep = _load_reputation()
    assert isinstance(rep, dict)
    assert "blacklist_name" in rep
    assert "blacklist_handle" in rep
    assert "trust_boost" in rep


def test_blacklisted_company_flagged():
    df = pd.DataFrame([
        {"title": "A", "company": "Skillfied Mentor",   "job_url": "http://ex.com/1"},
        {"title": "B", "company": "Anthropic",          "job_url": "http://ex.com/2"},
        {"title": "C", "company": "Random Real Corp",   "job_url": "http://ex.com/3"},
    ])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True
    assert bool(out.iloc[1]["pre_flagged_low_quality"]) is False
    assert bool(out.iloc[2]["pre_flagged_low_quality"]) is False


def test_trust_boost_company_gets_trusted_flag():
    df = pd.DataFrame([{"title": "X", "company": "Anthropic", "job_url": "https://anthropic.com/1"}])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_trusted"]) is True
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is False


def test_blacklisted_by_handle_in_url():
    """Even if company name is fine, a known-bad URL handle should flag the row."""
    df = pd.DataFrame([{
        "title": "A", "company": "Some Random Name",
        "job_url": "https://linkedin.com/posts/pankh-workforce-solution_x",
    }])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True


def test_case_insensitive_matching():
    df = pd.DataFrame([
        {"title": "A", "company": "SKILLFIED MENTOR LLP", "job_url": "x"},
        {"title": "B", "company": "skillfied mentor",     "job_url": "x"},
    ])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True
    assert bool(out.iloc[1]["pre_flagged_low_quality"]) is True


def test_empty_dataframe_survives():
    assert _pre_flag_reputation(pd.DataFrame()).empty
