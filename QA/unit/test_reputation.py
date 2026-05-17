"""Reputation prefilter: catches known job-mill companies before they reach AI."""
import pandas as pd
from pipeline.core_filter import _pre_flag_reputation, _load_reputation


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


# ---------------------------------------------------------------------------
# Companies added on 2026-05-17 after they leaked through into the daily email
# ---------------------------------------------------------------------------

def test_inficore_soft_blacklisted_after_internship_flood():
    """Inficore Soft posted 2 India-sus internships in the 2026-05-17 email."""
    df = pd.DataFrame([{"title": "Python Intern", "company": "Inficore Soft", "job_url": "x"}])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True


def test_skillzenloop_blacklisted_after_html_css_intern_flood():
    df = pd.DataFrame([{"title": "HTML/CSS Intern", "company": "Skillzenloop Pvt", "job_url": "x"}])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True


def test_netrolynx_blacklisted_after_data_analyst_leak():
    df = pd.DataFrame([{"title": "Data Analyst", "company": "Netrolynx AI", "job_url": "x"}])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True


def test_inficore_handle_pattern_in_linkedin_url():
    """The `inficore-soft` LinkedIn handle should also catch via URL even if the
    company field comes in differently spelled."""
    df = pd.DataFrame([{
        "title": "A", "company": "Generic Name",
        "job_url": "https://linkedin.com/company/inficore-soft/posts/xyz",
    }])
    out = _pre_flag_reputation(df).reset_index(drop=True)
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True
