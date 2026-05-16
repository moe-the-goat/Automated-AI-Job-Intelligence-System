"""core_notify rendering helpers: match cells, suspicious title prefix, sort, normalize_repo."""
import pandas as pd
from core_notify import (
    sort_by_match_percentage,
    _fmt_match_cell_html,
    _fmt_match_cell_md,
    _suspicious_title,
    _normalize_repo,
)


# --- Match cell formatting (HTML) ---

def test_match_cell_html_with_full_subscores():
    row = {"match_percentage": 75, "tech_fit": 92, "experience_fit": 65, "logistics_fit": 90}
    out = _fmt_match_cell_html(row)
    assert "<b>75%</b>" in out
    assert "T:92" in out
    assert "E:65" in out
    assert "L:90" in out


def test_match_cell_html_na_fallback():
    assert _fmt_match_cell_html({"match_percentage": "N/A"}) == "<b>N/A</b>"


# --- Match cell formatting (Markdown) ---

def test_match_cell_md_with_full_subscores():
    row = {"match_percentage": 75, "tech_fit": 92, "experience_fit": 65, "logistics_fit": 90}
    out = _fmt_match_cell_md(row)
    assert "**75%**" in out
    assert "T:92 E:65 L:90" in out


# --- Title badge precedence ---

def test_title_no_badge_when_clean():
    assert _suspicious_title("Data Intern", False) == "Data Intern"


def test_title_suspicious_badge():
    assert _suspicious_title("Data Intern", True) == "⚠️ Data Intern"


def test_title_blacklist_badge_outranks_suspicious():
    assert _suspicious_title("Data Intern", False, True) == "🚫 Data Intern"
    assert _suspicious_title("Data Intern", True, True) == "🚫 Data Intern"


def test_title_scam_badge_outranks_blacklist():
    assert _suspicious_title("Data Intern", False, False, True) == "🚨 Data Intern"
    assert _suspicious_title("Data Intern", False, True, True) == "🚨 Data Intern"
    assert _suspicious_title("Data Intern", True, True, True) == "🚨 Data Intern"


# --- Sort by match_percentage with tech_fit tiebreaker ---

def test_sort_uses_match_then_tech_tiebreaker():
    df = pd.DataFrame([
        {"title": "A", "match_percentage": 80, "tech_fit": 60},
        {"title": "B", "match_percentage": 80, "tech_fit": 95},
        {"title": "C", "match_percentage": 90, "tech_fit": 50},
    ])
    order = sort_by_match_percentage(df)["title"].tolist()
    assert order == ["C", "B", "A"]


def test_sort_empty_dataframe_returns_empty():
    assert sort_by_match_percentage(pd.DataFrame()).empty


# --- _normalize_repo ---

def test_normalize_repo_passthrough():
    assert _normalize_repo("moe-the-goat/job-scrapper-logs") == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_strips_https_prefix():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs"
    ) == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_strips_http_prefix():
    assert _normalize_repo(
        "http://github.com/moe-the-goat/job-scrapper-logs"
    ) == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_strips_trailing_slash():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs/"
    ) == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_strips_dot_git_suffix():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs.git"
    ) == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_handles_www_prefix():
    assert _normalize_repo(
        "https://www.github.com/moe-the-goat/job-scrapper-logs"
    ) == "moe-the-goat/job-scrapper-logs"


def test_normalize_repo_returns_none_for_blanks():
    assert _normalize_repo(None) is None
    assert _normalize_repo("") is None
    assert _normalize_repo("   ") is None
