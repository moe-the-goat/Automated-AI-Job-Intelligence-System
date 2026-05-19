import smtplib
import os
import re
import requests
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


def _normalize_repo(repo):
    """Coerce a repo value into `owner/name` form.

    Accepts: `owner/name`, `https://github.com/owner/name`,
    `https://github.com/owner/name/`, `github.com/owner/name`,
    with or without `.git` suffix. Returns None if the input is empty
    or unrecognizable.
    """
    if not repo:
        return None
    s = str(repo).strip().rstrip("/")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^(www\.)?github\.com/", "", s)
    if s.endswith(".git"):
        s = s[:-4]
    return s or None

# Title prefixes the bot creates. cleanup_old_github_issues matches any of these.
MANAGED_ISSUE_TITLE_PREFIXES = (
    "Automated AI Job Alerts",
    "Automated Job Alerts",
    "Local Companies Job Alerts",
    "Local Companies Scan",
)

"""
CORE NOTIFY MODULE
------------------
Handles the final presentation of our results. It formats the data into 
clean HTML for emails and Markdown for GitHub issues, sorts them intelligently 
by match percentage, and handles the actual dispatching via SMTP or GitHub API.
"""

def sort_by_match_percentage(df):
    """Sort by Match % descending, with tech_fit as the tiebreaker so equally-rated
    jobs surface the one with the stronger tech alignment first."""
    if df.empty or "match_percentage" not in df.columns:
        return df

    df = df.copy()
    df['_sort_pct'] = pd.to_numeric(df['match_percentage'].replace('N/A', -1), errors='coerce').fillna(-1)
    df['_sort_tech'] = pd.to_numeric(df.get('tech_fit', 0), errors='coerce').fillna(0)

    df = df.sort_values(by=['_sort_pct', '_sort_tech'], ascending=[False, False])
    df = df.drop(columns=['_sort_pct', '_sort_tech'])
    return df

def _match_pct_badge_html(pct):
    """Color-coded inline pill badge for an integer match percentage.

    Thresholds tuned for the rich-schema AI verdicts:
      >= 85  -> green   (#22c55e)  strong match — worth scanning first
      70-84  -> amber   (#eab308)  decent match — read carefully
      <  70  -> red     (#ef4444)  long shot or significant gap
      N/A    -> grey    (#9ca3af)  AI couldn't score (e.g. pre-screen skip)
    """
    try:
        n = int(pct) if str(pct).upper() != "N/A" else -1
    except (ValueError, TypeError):
        n = -1
    if n >= 85:
        color = "#22c55e"
    elif n >= 70:
        color = "#eab308"
    elif n >= 0:
        color = "#ef4444"
    else:
        color = "#9ca3af"
    label = f"{n}%" if n >= 0 else "N/A"
    return (
        f'<span style="background:{color};color:#fff;padding:3px 9px;'
        f'border-radius:6px;font-weight:bold;display:inline-block;'
        f'font-size:13px;">{label}</span>'
    )

def _fmt_match_cell_html(row):
    """Build the rich Match cell for HTML email: badge + sub-score breakdown."""
    pct = row.get("match_percentage", "N/A")
    badge = _match_pct_badge_html(pct)
    tech = row.get("tech_fit", "")
    exp = row.get("experience_fit", "")
    log = row.get("logistics_fit", "")
    if tech != "" and exp != "" and log != "":
        return f"{badge}<br><small>T:{tech} E:{exp} L:{log}</small>"
    return badge

def _fmt_match_cell_md(row):
    """Build the Match cell for Markdown (single line)."""
    pct = row.get("match_percentage", "N/A")
    pct_str = f"{pct}%" if str(pct).upper() != "N/A" else "N/A"
    tech = row.get("tech_fit", "")
    exp = row.get("experience_fit", "")
    log = row.get("logistics_fit", "")
    if tech != "" and exp != "" and log != "":
        return f"**{pct_str}** (T:{tech} E:{exp} L:{log})"
    return f"**{pct_str}**"

# Anchored markers the AI prompt instructs the model to use in its verdict.
# Bolding them in the rendered output makes the structure scannable at a glance.
_VERDICT_KEYWORD_RE = re.compile(
    r"\b(MATCH:|GAP:|CLOSING[\s\-]REASON:|CLOSING:|REASON:)",
    re.IGNORECASE,
)

def _bolden_verdict_html(verdict):
    """Wrap MATCH:/GAP:/CLOSING-REASON: markers in <strong> for HTML output."""
    if not verdict:
        return ""
    return _VERDICT_KEYWORD_RE.sub(r"<strong>\1</strong>", str(verdict))

def _bolden_verdict_md(verdict):
    """Same but with markdown bold syntax. Caller still escapes pipes for tables."""
    if not verdict:
        return ""
    return _VERDICT_KEYWORD_RE.sub(r"**\1**", str(verdict))

def _suspicious_title(title, is_suspicious, is_blacklisted=False, is_scam=False):
    """Prepend a visible warning to titles. Severity order (most severe first):

    🚨 = web-confirmed scam (India + suspicious + Reddit/review mentions of fraud).
    🚫 = pre-flagged low-quality via the reputation list (A1 / data/reputation.json).
    ⚠️  = AI-flagged suspicious posting (the model's own judgement).
    """
    if is_scam:
        return f"🚨 {title}"
    if is_blacklisted:
        return f"🚫 {title}"
    if is_suspicious:
        return f"⚠️ {title}"
    return title

def _render_html_table(df):
    """Render one section's table with the rich schema."""
    out = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    out += (
        "<tr>"
        "<th>Title</th><th>Company</th><th>Location</th>"
        "<th>Match</th><th>Pay</th><th>Effort</th>"
        "<th>AI Verdict</th><th>Link</th>"
        "</tr>"
    )
    for _, row in df.iterrows():
        title = _suspicious_title(
            row.get("title", "N/A"),
            bool(row.get("suspicious", False)),
            bool(row.get("pre_flagged_low_quality", False)),
            bool(row.get("scam", False)),
        )
        company = row.get("company", "N/A")
        location = row.get("location", "Remote/Unspecified")
        match_cell = _fmt_match_cell_html(row)
        comp = row.get("compensation", "Not stated")
        effort = row.get("effort", "unknown")
        verdict = _bolden_verdict_html(row.get("ai_verdict", ""))
        job_url = row.get("job_url", "#")
        out += (
            f"<tr><td>{title}</td><td>{company}</td><td>{location}</td>"
            f"<td>{match_cell}</td><td>{comp}</td><td>{effort}</td>"
            f"<td>{verdict}</td><td><a href='{job_url}'>Apply</a></td></tr>"
        )
    out += "</table>"
    return out

def _geo_title_prefix(row):
    """Return a small prefix indicating geo-eligibility status for the lower-ranked section.

    Only the "uncertain" state gets a visible warning marker — "open" and unchecked
    jobs render cleanly so the section stays scannable. "restricted" rows never
    reach the renderer (scraper.py drops them before sending the lower_ranked_df in).
    """
    status = (row.get("geo_status") or "").lower()
    if status == "uncertain":
        return "⚠️ "
    return ""


def _render_lower_ranked_html(df, limit=25):
    """Compact table for jobs that didn't make the AI top-N: title, company, sim score, link."""
    df = df.copy()
    if "similarity" in df.columns:
        df = df.sort_values("similarity", ascending=False)
    df = df.head(limit)
    out = "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    out += "<tr><th>Title</th><th>Company</th><th>Location</th><th>Similarity</th><th>Link</th></tr>"
    for _, row in df.iterrows():
        title_prefix = _geo_title_prefix(row)
        title = f"{title_prefix}{row.get('title', 'N/A')}"
        company = row.get("company", "N/A")
        location = row.get("location", "Remote/Unspecified")
        sim = row.get("similarity", 0.0)
        sim_str = f"{float(sim):.2f}" if sim is not None else "—"
        job_url = row.get("job_url", "#")
        out += (
            f"<tr><td>{title}</td><td>{company}</td><td>{location}</td>"
            f"<td>{sim_str}</td><td><a href='{job_url}'>View</a></td></tr>"
        )
    out += "</table>"
    return out


def _render_lower_ranked_md(df, limit=25):
    df = df.copy()
    if "similarity" in df.columns:
        df = df.sort_values("similarity", ascending=False)
    df = df.head(limit)
    out = "| Title | Company | Location | Similarity | Link |\n"
    out += "|---|---|---|---|---|\n"
    for _, row in df.iterrows():
        title_prefix = _geo_title_prefix(row)
        title = f"{title_prefix}{row.get('title', 'N/A')}"
        company = row.get("company", "N/A")
        location = row.get("location", "Remote/Unspecified")
        sim = row.get("similarity", 0.0)
        sim_str = f"{float(sim):.2f}" if sim is not None else "—"
        job_url = row.get("job_url", "#")
        out += f"| {title} | {company} | {location} | {sim_str} | [View]({job_url}) |\n"
    return out


def format_email_html(internships_df, jobs_df, stats, lower_ranked_df=None):
    """Generates the final HTML payload for the daily email.

    `lower_ranked_df` (A3) is an optional dataframe of jobs that survived
    filtering but didn't make the AI top-N — rendered as a compact summary
    so the long tail is visible without burning AI quota.
    """
    internships_df = sort_by_match_percentage(internships_df.copy() if not internships_df.empty else pd.DataFrame())
    jobs_df = sort_by_match_percentage(jobs_df.copy() if not jobs_df.empty else pd.DataFrame())

    html = "<h2>Automated AI Job Alerts</h2>"
    html += f"<div><b>Pipeline Stats:</b> Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: {stats['approved']}</div>"
    html += (
        "<div style='color: #666; font-size: 12px; margin-top:4px;'>"
        "Match cell shows composite % (color-coded: "
        "<span style='background:#22c55e;color:#fff;padding:1px 6px;border-radius:4px;'>&ge;85</span> "
        "<span style='background:#eab308;color:#fff;padding:1px 6px;border-radius:4px;'>70-84</span> "
        "<span style='background:#ef4444;color:#fff;padding:1px 6px;border-radius:4px;'>&lt;70</span>) "
        "with sub-scores Tech / Experience / Logistics. "
        "🚨 = web-confirmed scam · 🚫 = blacklisted company · ⚠️ = AI-flagged suspicious.</div><hr>"
    )

    html += "<h3>🎓 Internships (AI & SWE)</h3>"
    if internships_df.empty:
        html += "<p>No relevant internships found today.</p>"
    else:
        html += _render_html_table(internships_df) + "<br>"

    html += "<h3>💼 Junior & Entry-Level Jobs (CV-Matched)</h3>"
    if jobs_df.empty:
        html += "<p>No relevant full-time jobs found today.</p>"
    else:
        html += _render_html_table(jobs_df)

    if lower_ranked_df is not None and not lower_ranked_df.empty:
        html += f"<br><h3>📋 Lower-Ranked Matches ({len(lower_ranked_df)} jobs — no AI verdict)</h3>"
        html += "<div style='color: #666; font-size: 12px;'>These passed the deterministic filters but ranked below the top-N by CV similarity, so AI didn't evaluate them. Higher similarity = closer to your CV.</div>"
        html += _render_lower_ranked_html(lower_ranked_df)

    return html

def send_email(subject, html_content, email_settings):
    """Dispatches the HTML email via Gmail SMTP."""
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    
    if not sender_email or not app_password:
        logger.error("SENDER_EMAIL or EMAIL_APP_PASSWORD environment variables are not set.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(email_settings["receiver_emails"])

    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP_SSL(email_settings["smtp_server"], email_settings["smtp_port"]) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error("Failed to send email: %s", e)

def _render_md_table(df):
    """Render one section's table in Markdown with the rich schema."""
    out = "| Title | Company | Location | Match | Pay | Effort | AI Verdict | Link |\n"
    out += "|---|---|---|---|---|---|---|---|\n"
    for _, row in df.iterrows():
        title = _suspicious_title(
            row.get("title", "N/A"),
            bool(row.get("suspicious", False)),
            bool(row.get("pre_flagged_low_quality", False)),
            bool(row.get("scam", False)),
        )
        company = row.get("company", "N/A")
        location = row.get("location", "Remote/Unspecified")
        match_cell = _fmt_match_cell_md(row)
        comp = row.get("compensation", "Not stated")
        effort = row.get("effort", "unknown")
        verdict = _bolden_verdict_md(row.get("ai_verdict", ""))
        job_url = row.get("job_url", "#")
        # Escape pipes inside cells so the markdown table doesn't break.
        verdict = str(verdict).replace("|", "\\|")
        out += f"| {title} | {company} | {location} | {match_cell} | {comp} | {effort} | {verdict} | [Apply]({job_url}) |\n"
    return out

def format_github_markdown(internships_df, jobs_df, stats, lower_ranked_df=None):
    """Generates Markdown formatting for a GitHub Issue payload."""
    internships_df = sort_by_match_percentage(internships_df.copy() if not internships_df.empty else pd.DataFrame())
    jobs_df = sort_by_match_percentage(jobs_df.copy() if not jobs_df.empty else pd.DataFrame())

    md = "## Automated AI Job Alerts\n\n"
    md += f"**Pipeline Stats:** Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: {stats['approved']}\n\n"
    md += "_Match shows composite % with sub-scores Tech / Experience / Logistics. 🚨 = web-confirmed scam · 🚫 = blacklisted · ⚠️ = AI-suspicious._\n\n---\n\n"

    md += "### 🎓 Internships (AI & SWE)\n\n"
    if internships_df.empty:
        md += "No relevant internships found today.\n\n"
    else:
        md += _render_md_table(internships_df) + "\n"

    md += "### 💼 Junior & Entry-Level Jobs (CV-Matched)\n\n"
    if jobs_df.empty:
        md += "No relevant full-time jobs found today.\n\n"
    else:
        md += _render_md_table(jobs_df)

    if lower_ranked_df is not None and not lower_ranked_df.empty:
        md += f"\n\n### 📋 Lower-Ranked Matches ({len(lower_ranked_df)} jobs — no AI verdict)\n\n"
        md += "_Passed the deterministic filters but ranked below the top-N by CV similarity, so AI didn't evaluate them._\n\n"
        md += _render_lower_ranked_md(lower_ranked_df)

    return md

def create_github_issue(title, body, repo=None, token=None):
    """Posts a new GitHub Issue with the collected jobs.

    `repo` (in `owner/name` form) and `token` (a PAT or GitHub Actions token)
    can be passed explicitly to route the issue to a different repository
    (e.g. the private logs repo). When omitted they fall back to the
    `GITHUB_REPOSITORY` and `GITHUB_TOKEN` environment variables that
    GitHub Actions sets automatically.
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    repo = _normalize_repo(repo or os.environ.get("GITHUB_REPOSITORY"))

    if not token or not repo:
        logger.error("No GitHub token/repo available for issue creation.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"title": title, "body": body}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logger.info("GitHub Issue created in %s: %s", repo, response.json().get('html_url'))
    except Exception as e:
        logger.error("Failed to create GitHub Issue in %s: %s", repo, e)

def cleanup_old_github_issues(days_old=5, repo=None, token=None):
    """Closes managed bot-issues older than `days_old` calendar days.

    `repo` and `token` follow the same override pattern as `create_github_issue`:
    explicit args win; otherwise fall back to GITHUB_REPOSITORY / GITHUB_TOKEN.
    Lets the caller sweep the private logs repo AND the legacy public repo
    by calling this function twice with different `(repo, token)` pairs.
    """
    token = token or os.environ.get("GITHUB_TOKEN")
    repo = _normalize_repo(repo or os.environ.get("GITHUB_REPOSITORY"))

    if not token or not repo:
        return

    url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        issues = response.json()

        now = datetime.now(timezone.utc)
        closed_count = 0
        for issue in issues:
            title = issue.get("title", "")
            if not any(title.startswith(prefix) for prefix in MANAGED_ISSUE_TITLE_PREFIXES):
                continue
            created_at_str = issue.get("created_at")
            if not created_at_str:
                continue
            created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            # Compare CALENDAR days, not 24-hour periods. An issue from May 8 should
            # count as 5 days old on May 13 regardless of clock time.
            age_days = (now.date() - created_at.date()).days
            if age_days >= days_old:
                issue_num = issue.get("number")
                logger.info("Closing old issue #%s '%s' in %s (Age: %d days)", issue_num, title, repo, age_days)
                patch_url = f"https://api.github.com/repos/{repo}/issues/{issue_num}"
                requests.patch(patch_url, headers=headers, json={"state": "closed"})
                closed_count += 1
        logger.info("GitHub cleanup done in %s. Closed %d stale issue(s).", repo, closed_count)

    except Exception as e:
        logger.error("Failed to cleanup old GitHub issues: %s", e)
