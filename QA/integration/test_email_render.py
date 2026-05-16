"""End-to-end rendering of format_email_html: realistic dataframes -> HTML.

Exercises the full layering: sort -> table rendering -> match-cell sub-scores ->
suspicious/blacklist/scam title badges -> compensation column -> effort column ->
optional lower-ranked section.
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fixtures"))
from sample_jobs import baseline_job_row
from core_notify import format_email_html, format_github_markdown


def _stats(approved=2):
    return {"scraped": 100, "filtered": 50, "approved": approved}


def test_email_includes_pipeline_stats():
    df = pd.DataFrame([baseline_job_row()])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "Scraped: 100" in html
    assert "Filtered to: 50" in html
    assert "AI Approved: 1" in html


def test_email_renders_internships_table():
    df = pd.DataFrame([
        baseline_job_row(title="AI Engineering Intern", company="iion"),
        baseline_job_row(title="Data Science Intern", company="OtherCo"),
    ])
    html = format_email_html(df, pd.DataFrame(), _stats(2))
    assert "AI Engineering Intern" in html
    assert "Data Science Intern" in html
    assert "Internships" in html


def test_email_renders_subscores_in_match_cell():
    df = pd.DataFrame([baseline_job_row(
        match_percentage=82, tech_fit=95, experience_fit=70, logistics_fit=85)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "<b>82%</b>" in html
    assert "T:95" in html
    assert "E:70" in html
    assert "L:85" in html


def test_email_renders_compensation_and_effort_columns():
    df = pd.DataFrame([baseline_job_row(compensation="$30/hr", effort="medium")])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "$30/hr" in html
    assert ">medium<" in html


def test_email_renders_suspicious_badge():
    df = pd.DataFrame([baseline_job_row(title="Suspicious Role", suspicious=True)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "⚠️ Suspicious Role" in html


def test_email_renders_blacklist_badge():
    df = pd.DataFrame([baseline_job_row(
        title="Lazy Repost", pre_flagged_low_quality=True)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "🚫 Lazy Repost" in html


def test_email_renders_scam_badge_outranks_others():
    df = pd.DataFrame([baseline_job_row(
        title="Fake Pvt Ltd Role", suspicious=True,
        pre_flagged_low_quality=True, scam=True)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "🚨 Fake Pvt Ltd Role" in html
    # Blacklist + suspicious badges should NOT also appear on the same title
    assert "🚫 Fake" not in html
    assert "⚠️ Fake" not in html


def test_email_renders_lower_ranked_section_when_provided():
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "Low-sim 1", "company": "X", "location": "R",
         "similarity": 0.45, "job_url": "https://example.com/lr1"},
        {"title": "Low-sim 2", "company": "Y", "location": "R",
         "similarity": 0.32, "job_url": "https://example.com/lr2"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    assert "Lower-Ranked Matches" in html
    assert "2 jobs" in html
    assert "Low-sim 1" in html
    assert "0.45" in html


def test_email_omits_lower_ranked_section_when_empty_or_none():
    main = pd.DataFrame([baseline_job_row()])
    html_none = format_email_html(main, pd.DataFrame(), _stats(1))
    html_empty = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=pd.DataFrame())
    assert "Lower-Ranked Matches" not in html_none
    assert "Lower-Ranked Matches" not in html_empty


def test_email_uses_no_jobs_placeholders():
    """When both dataframes are empty, the email still renders with placeholder paragraphs."""
    html = format_email_html(pd.DataFrame(), pd.DataFrame(), _stats(0))
    assert "No relevant internships" in html
    assert "No relevant full-time jobs" in html


def test_markdown_renders_match_cell_with_subscores():
    df = pd.DataFrame([baseline_job_row(
        match_percentage=92, tech_fit=95, experience_fit=90, logistics_fit=90)])
    md = format_github_markdown(df, pd.DataFrame(), _stats(1))
    assert "**92%** (T:95 E:90 L:90)" in md


def test_markdown_escapes_pipes_in_verdict():
    """A verdict with a literal `|` character would otherwise break the markdown table."""
    df = pd.DataFrame([baseline_job_row(ai_verdict="uses pipe | inside text")])
    md = format_github_markdown(df, pd.DataFrame(), _stats(1))
    assert "uses pipe \\| inside text" in md


def test_markdown_renders_legend_with_all_three_badges():
    md = format_github_markdown(pd.DataFrame(), pd.DataFrame(), _stats(0))
    assert "🚨" in md
    assert "🚫" in md
    assert "⚠️" in md
