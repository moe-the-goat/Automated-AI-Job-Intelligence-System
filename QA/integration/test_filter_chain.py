"""apply_pipeline_filters end-to-end on a diverse 20-row dataframe.

This is the closest thing to "run the whole pre-AI gauntlet" without making
network calls. It catches regressions in any of: seen-jobs filter, reputation
prefilter, URL dedup, title+company dedup, CJK reject, langdetect reject,
location prefilter, seniority reject, tech-keyword filter.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fixtures"))
from sample_jobs import mixed_filter_input
from pipeline.core_filter import apply_pipeline_filters


def test_filter_drops_senior_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).tolist()
    assert not any("Senior" in t for t in titles)
    assert not any("Lead Backend" in t for t in titles)
    assert not any("Engineering Manager" in t for t in titles)
    assert not any("Staff AI" in t for t in titles)


def test_filter_drops_non_tech_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).str.lower().tolist()
    assert not any("marketing coordinator" in t for t in titles)
    assert not any("sales operations manager" in t for t in titles)


def test_filter_drops_cjk_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).tolist()
    assert "ソフトウェアエンジニア" not in titles


def test_filter_drops_us_state_locked_when_title_lacks_remote():
    out = apply_pipeline_filters(mixed_filter_input())
    rows = out[out["company"] == "USCo"]
    assert len(rows) == 0   # location "San Francisco, California" without "Remote" in title -> dropped


def test_filter_keeps_us_state_when_title_has_remote():
    out = apply_pipeline_filters(mixed_filter_input())
    rows = out[out["company"] == "USCo2"]
    assert len(rows) == 1   # title "Remote Software Engineer" overrides location filter


def test_filter_flags_blacklisted_but_does_not_drop():
    """Reputation-blacklisted jobs are flagged via pre_flagged_low_quality, NOT dropped."""
    out = apply_pipeline_filters(mixed_filter_input())
    flagged = out[out["pre_flagged_low_quality"] == True]
    flagged_companies = flagged["company"].astype(str).str.lower().tolist()
    assert any("skillfied" in c for c in flagged_companies)
    assert any("webs it" in c for c in flagged_companies)


def test_filter_tags_trusted_company():
    """Anthropic is in the trust_boost list and should carry the trusted flag."""
    out = apply_pipeline_filters(mixed_filter_input())
    trusted = out[out["pre_flagged_trusted"] == True]
    trusted_companies = trusted["company"].astype(str).str.lower().tolist()
    assert "anthropic" in trusted_companies


def test_filter_deduplicates_url_collisions():
    """Two rows with identical job_url should collapse to one."""
    out = apply_pipeline_filters(mixed_filter_input())
    url_counts = out["job_url"].value_counts()
    assert url_counts.max() == 1


def test_filter_deduplicates_normalized_title_company():
    """'AI Engineer' vs 'AI Engineer (Remote)' for the same company should collapse."""
    out = apply_pipeline_filters(mixed_filter_input())
    same_co = out[out["company"].astype(str).str.lower() == "iion"]
    # At most one of: 'AI Engineering Intern' vs 'AI Engineering Intern (Remote)'
    assert len(same_co) <= 1


def test_filter_survives_empty_dataframe():
    import pandas as pd
    assert apply_pipeline_filters(pd.DataFrame()).empty
