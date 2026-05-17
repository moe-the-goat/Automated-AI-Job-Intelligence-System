"""core_notify rendering helpers: match cells, suspicious title prefix, sort, normalize_repo."""
import pandas as pd
from pipeline.core_notify import (
    sort_by_match_percentage,
    _fmt_match_cell_html,
    _fmt_match_cell_md,
    _suspicious_title,
    _normalize_repo,
    _match_pct_badge_html,
    _bolden_verdict_html,
    _bolden_verdict_md,
)


# --- Match cell formatting (HTML) — now renders a color-coded badge ---

def test_match_cell_html_with_full_subscores():
    row = {"match_percentage": 75, "tech_fit": 92, "experience_fit": 65, "logistics_fit": 90}
    out = _fmt_match_cell_html(row)
    # Badge contains the percentage label inside a styled span.
    assert "75%" in out
    assert "background:" in out                                # inline color style
    assert "T:92" in out
    assert "E:65" in out
    assert "L:90" in out


def test_match_cell_html_na_fallback():
    """N/A still produces a (grey) badge so the visual layout stays consistent."""
    out = _fmt_match_cell_html({"match_percentage": "N/A"})
    assert "N/A" in out
    assert "<span" in out                                       # rendered as a badge


# --- Match-percentage badge color thresholds (Fix #4) ---

def test_badge_green_at_high_match():
    """>=85 should produce a green (#22c55e) badge."""
    out = _match_pct_badge_html(95)
    assert "#22c55e" in out
    assert "95%" in out


def test_badge_green_at_threshold_85():
    out = _match_pct_badge_html(85)
    assert "#22c55e" in out


def test_badge_yellow_at_mid_match():
    """70-84 should produce an amber (#eab308) badge."""
    out = _match_pct_badge_html(78)
    assert "#eab308" in out
    assert "78%" in out


def test_badge_yellow_at_threshold_70():
    out = _match_pct_badge_html(70)
    assert "#eab308" in out


def test_badge_red_at_low_match():
    """<70 should produce a red (#ef4444) badge."""
    out = _match_pct_badge_html(45)
    assert "#ef4444" in out
    assert "45%" in out


def test_badge_red_at_zero():
    out = _match_pct_badge_html(0)
    assert "#ef4444" in out
    assert "0%" in out


def test_badge_grey_for_na_input():
    out = _match_pct_badge_html("N/A")
    assert "#9ca3af" in out
    assert "N/A" in out


def test_badge_handles_string_number():
    """Some pipelines coerce to string before render — should still color correctly."""
    out = _match_pct_badge_html("88")
    assert "#22c55e" in out


def test_badge_handles_malformed_input():
    """Garbage input should fall back to grey, not crash."""
    out = _match_pct_badge_html("not-a-number")
    assert "#9ca3af" in out


# --- Verdict bolding (Fix #4) ---

def test_bolden_html_wraps_match_keyword():
    out = _bolden_verdict_html("MATCH: Strong Python + PyTorch overlap. GAP: 2 yrs prod ML experience needed.")
    assert "<strong>MATCH:</strong>" in out
    assert "<strong>GAP:</strong>" in out


def test_bolden_html_wraps_closing_reason():
    out = _bolden_verdict_html("MATCH: x. GAP: y. CLOSING-REASON: worth applying.")
    assert "<strong>CLOSING-REASON:</strong>" in out


def test_bolden_html_handles_closing_reason_with_space_or_hyphen():
    # Tolerance for both 'CLOSING-REASON:' and 'CLOSING REASON:' spellings.
    out = _bolden_verdict_html("Some text. CLOSING REASON: yes.")
    assert "<strong>CLOSING REASON:</strong>" in out


def test_bolden_html_passes_through_text_without_keywords():
    """If no keywords match, the verdict comes back unchanged."""
    out = _bolden_verdict_html("Plain verdict text with no markers.")
    assert out == "Plain verdict text with no markers."


def test_bolden_html_handles_empty_and_none():
    assert _bolden_verdict_html("") == ""
    assert _bolden_verdict_html(None) == ""


def test_bolden_md_wraps_with_markdown_bold():
    out = _bolden_verdict_md("MATCH: Strong Python. GAP: prod ML experience.")
    assert "**MATCH:**" in out
    assert "**GAP:**" in out


def test_bolden_md_does_not_use_html_tags():
    """Markdown variant must not emit <strong>."""
    out = _bolden_verdict_md("MATCH: x")
    assert "<strong>" not in out
    assert "**MATCH:**" in out


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
