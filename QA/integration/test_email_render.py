"""End-to-end rendering of format_email_html: realistic dataframes -> HTML.

Exercises the full layering: sort -> table rendering -> match-cell sub-scores ->
suspicious/blacklist/scam title badges -> compensation column -> effort column ->
optional lower-ranked section.
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fixtures"))
from sample_jobs import baseline_job_row
from pipeline.core_notify import format_email_html, format_github_markdown


def _stats(approved=2):
    return {"scraped": 100, "filtered": 50, "approved": approved}


def test_email_includes_pipeline_stats():
    df = pd.DataFrame([baseline_job_row()])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "Scraped: 100" in html
    assert "Filtered to: 50" in html
    assert "AI Approved: 1" in html


def test_email_approved_count_matches_displayed_rows_not_raw_total():
    """Regression: the 'AI Approved' stat must count the jobs the email actually
    shows, not the caller's raw pre-threshold total. A job that cleared the AI
    (is_valid) but scored below the display floor (default 55) is dropped from
    the tables, so it must not be counted — otherwise the stat reads '3' while
    the reader sees 2 (the bug observed in production)."""
    df = pd.DataFrame([
        baseline_job_row(title="Strong A", match_percentage=86),
        baseline_job_row(title="Strong B", match_percentage=85),
        baseline_job_row(title="Weak C", match_percentage=49),  # below the 55 floor
    ])
    # Caller passes the raw approved total (3) — the renderer must correct it.
    html = format_email_html(df, pd.DataFrame(), _stats(3))
    assert "AI Approved: 2" in html
    assert "AI Approved: 3" not in html
    # And the hidden job is genuinely absent from the body.
    assert "Strong A" in html
    assert "Strong B" in html
    assert "Weak C" not in html


def test_markdown_approved_count_matches_displayed_rows():
    """Same correction for the GitHub-Issue markdown output."""
    df = pd.DataFrame([
        baseline_job_row(title="Strong A", match_percentage=86),
        baseline_job_row(title="Weak C", match_percentage=49),
    ])
    md = format_github_markdown(df, pd.DataFrame(), _stats(2))
    assert "AI Approved: 1" in md
    assert "Weak C" not in md


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
    # 82 falls in the 70-84 (yellow / amber) bracket — verify the colored badge
    # is present and the sub-scores render.
    assert "82%" in html
    assert "#eab308" in html                                    # amber badge for 70-84
    assert "T:95" in html
    assert "E:70" in html
    assert "L:85" in html


def test_email_renders_green_badge_for_high_match():
    """A 90% match should produce a green badge (#22c55e) in the rendered email."""
    df = pd.DataFrame([baseline_job_row(match_percentage=90)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "#22c55e" in html
    assert "90%" in html


def test_email_renders_red_badge_for_low_match():
    """A 55% match should produce a red badge (#ef4444)."""
    df = pd.DataFrame([baseline_job_row(match_percentage=55)])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "#ef4444" in html


def test_email_bolds_verdict_match_and_gap_keywords():
    """MATCH: and GAP: in the AI verdict should render as <strong> for scannability."""
    df = pd.DataFrame([baseline_job_row(
        ai_verdict="MATCH: Strong Python + PyTorch. GAP: 2 yrs prod ML experience."
    )])
    html = format_email_html(df, pd.DataFrame(), _stats(1))
    assert "<strong>MATCH:</strong>" in html
    assert "<strong>GAP:</strong>" in html


def test_email_header_includes_color_legend():
    """The stats header should explain the color coding to first-time readers."""
    html = format_email_html(pd.DataFrame(), pd.DataFrame(), _stats(0))
    assert "color-coded" in html
    assert "#22c55e" in html                                    # legend swatch
    assert "#eab308" in html
    assert "#ef4444" in html


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


def test_email_renders_also_found_section_when_provided():
    """Lower-ranked section renders as 'Also Found' with weighted_score as percentage."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "Low-sim 1", "company": "X", "location": "R",
         "similarity": 0.45, "weighted_score": 0.45,
         "job_url": "https://example.com/lr1"},
        {"title": "Low-sim 2", "company": "Y", "location": "R",
         "similarity": 0.32, "weighted_score": 0.32,
         "job_url": "https://example.com/lr2"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    assert "Also Found" in html
    assert "2 jobs" in html
    assert "Low-sim 1" in html
    # weighted_score 0.45 -> "45%"
    assert "45%" in html


def test_email_omits_also_found_section_when_empty_or_none():
    main = pd.DataFrame([baseline_job_row()])
    html_none = format_email_html(main, pd.DataFrame(), _stats(1))
    html_empty = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=pd.DataFrame())
    assert "Also Found" not in html_none
    assert "Also Found" not in html_empty


def test_email_also_found_filters_blacklisted_rows():
    """Reputation-flagged rows should NOT appear in Also Found — they were already
    flagged in the AI section (or pre-screen-skipped) and showing them twice clutters
    the view with the same Inficore/Skillzenloop/Techskill spam."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "Clean Job", "company": "RealCo", "location": "Berlin",
         "similarity": 0.50, "weighted_score": 0.58,
         "pre_flagged_low_quality": False,
         "job_url": "https://example.com/clean"},
        {"title": "Spam Job", "company": "Inficore Soft", "location": "Bangalore",
         "similarity": 0.60, "weighted_score": 0.42,
         "pre_flagged_low_quality": True,
         "job_url": "https://example.com/spam"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    assert "Clean Job" in html
    assert "Spam Job" not in html
    # The count in the header should reflect the post-filter count (1, not 2)
    assert "1 jobs" in html


def test_email_also_found_caps_at_15_rows():
    """Section should hard-cap at 15 rows even if many lower-ranked jobs survive filtering."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": f"Row {i:02d}", "company": "Co", "location": "Remote",
         "similarity": 0.5 - i * 0.001, "weighted_score": 0.6 - i * 0.001,
         "pre_flagged_low_quality": False,
         "job_url": f"https://example.com/{i}"}
        for i in range(30)
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    assert "15 jobs" in html
    # First 15 should appear (sorted by weighted_score desc -> Row 00..14)
    assert "Row 00" in html
    assert "Row 14" in html
    # Row 15+ should be dropped
    assert "Row 15" not in html


def test_email_also_found_sorts_by_weighted_score_not_similarity():
    """A row with HIGHER similarity but LOWER weighted_score (e.g. India-deweighted)
    should appear BELOW a row with lower similarity but higher weighted (e.g. EU-boosted)."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "High Sim Low Weight", "company": "X", "location": "Bangalore",
         "similarity": 0.65, "weighted_score": 0.46,
         "pre_flagged_low_quality": False,
         "job_url": "https://example.com/a"},
        {"title": "Low Sim High Weight", "company": "Y", "location": "Berlin",
         "similarity": 0.50, "weighted_score": 0.58,
         "pre_flagged_low_quality": False,
         "job_url": "https://example.com/b"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    # "Low Sim High Weight" should appear FIRST in the HTML (higher in the table)
    assert html.index("Low Sim High Weight") < html.index("High Sim Low Weight")


def test_email_renders_also_found_with_ai_verdicts():
    """When the lower-ranked frame carries AI eval columns, the Also Found
    section uses the Match-badge + AI Verdict layout (not just the legacy
    Score percentage)."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "Gemini-eval job", "company": "EuCo", "location": "Berlin",
         "similarity": 0.55, "weighted_score": 0.63,
         "match_percentage": 78, "tech_fit": 80, "experience_fit": 70, "logistics_fit": 80,
         "compensation": "Not stated", "effort": "medium",
         "suspicious": False, "scam": False, "pre_flagged_low_quality": False,
         "ai_verdict": "MATCH: Your RAG project is a direct fit.",
         "job_url": "https://example.com/lr-ai"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    assert "Also Found" in html
    assert "Gemini-evaluated" in html
    assert "Gemini-eval job" in html
    # Match badge with the score:
    assert "78%" in html
    # Sub-scores rendered like the top section:
    assert "T:80" in html
    # AI verdict text shows up with MATCH bolded:
    assert "<strong>MATCH:</strong>" in html
    # Match cell header used (not the legacy "Score" header):
    assert "<th>Match</th>" in html
    assert "<th>AI Verdict</th>" in html


def test_email_also_found_sorts_by_match_percentage_when_ai_eval_present():
    """With AI columns the sort key switches from weighted_score to match_percentage."""
    main = pd.DataFrame([baseline_job_row()])
    lower = pd.DataFrame([
        {"title": "Lower match", "company": "X", "location": "R",
         "similarity": 0.55, "weighted_score": 0.70,
         "match_percentage": 60, "tech_fit": 60, "experience_fit": 60, "logistics_fit": 60,
         "compensation": "n/a", "effort": "low",
         "suspicious": False, "scam": False, "pre_flagged_low_quality": False,
         "ai_verdict": "Generic fit.",
         "job_url": "https://example.com/lower-match"},
        {"title": "Higher match", "company": "Y", "location": "R",
         "similarity": 0.50, "weighted_score": 0.58,
         "match_percentage": 75, "tech_fit": 75, "experience_fit": 75, "logistics_fit": 75,
         "compensation": "n/a", "effort": "low",
         "suspicious": False, "scam": False, "pre_flagged_low_quality": False,
         "ai_verdict": "Stronger fit.",
         "job_url": "https://example.com/higher-match"},
    ])
    html = format_email_html(main, pd.DataFrame(), _stats(1), lower_ranked_df=lower)
    # "Higher match" (75%) sorts above "Lower match" (60%) despite lower weighted_score.
    assert html.index("Higher match") < html.index("Lower match")


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
