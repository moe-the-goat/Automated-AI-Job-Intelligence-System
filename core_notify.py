import smtplib
import os
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

"""
CORE NOTIFY MODULE
------------------
Handles the final presentation of our results. It formats the data into 
clean HTML for emails and Markdown for GitHub issues, sorts them intelligently 
by match percentage, and handles the actual dispatching via SMTP or GitHub API.
"""

def sort_by_match_percentage(df):
    """Sorts jobs so the highest Match % appears at the top. (Tier 3 Item 13)"""
    if df.empty or "match_percentage" not in df.columns:
        return df
    
    # Temporarily replace 'N/A' or missing with -1 to sort them to the bottom
    df['sort_pct'] = df['match_percentage'].replace('N/A', -1)
    df['sort_pct'] = pd.to_numeric(df['sort_pct'], errors='coerce').fillna(-1)
    
    df = df.sort_values(by=['sort_pct'], ascending=False)
    df = df.drop(columns=['sort_pct'])
    return df

def format_email_html(internships_df, jobs_df, stats):
    """Generates the final HTML payload for the daily email."""
    # Sort the dataframes first
    import pandas as pd
    internships_df = sort_by_match_percentage(internships_df.copy() if not internships_df.empty else pd.DataFrame())
    jobs_df = sort_by_match_percentage(jobs_df.copy() if not jobs_df.empty else pd.DataFrame())

    html = "<h2>Automated AI Job Alerts</h2>"
    
    # --- Daily Stats Summary (Tier 3 Item 14) ---
    html += f"<div><b>Pipeline Stats:</b> Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: {stats['approved']}</div><hr>"
    
    # --- Internships Table ---
    html += "<h3>🎓 Internships (AI & SWE)</h3>"
    if internships_df.empty:
        html += "<p>No relevant internships found today.</p>"
    else:
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Title</th><th>Company</th><th>Location</th><th>AI Verdict</th><th>Match %</th><th>Link</th></tr>"
        for _, row in internships_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if str(match_pct) != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            html += f"<tr><td>{title}</td><td>{company}</td><td>{location}</td><td>{verdict}</td><td>{match_pct}</td><td><a href='{job_url}'>Apply</a></td></tr>"
        html += "</table><br>"
        
    # --- Full-Time Jobs Table ---
    html += "<h3>💼 Junior & Entry-Level Jobs (CV-Matched)</h3>"
    if jobs_df.empty:
        html += "<p>No relevant full-time jobs found today.</p>"
    else:
        html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html += "<tr><th>Title</th><th>Company</th><th>Location</th><th>AI Verdict</th><th>Match %</th><th>Link</th></tr>"
        for _, row in jobs_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if str(match_pct) != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            html += f"<tr><td>{title}</td><td>{company}</td><td>{location}</td><td>{verdict}</td><td>{match_pct}</td><td><a href='{job_url}'>Apply</a></td></tr>"
        html += "</table>"
        
    return html

def send_email(subject, html_content, email_settings):
    """Dispatches the HTML email via Gmail SMTP."""
    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")
    
    if not sender_email or not app_password:
        print("Error: SENDER_EMAIL or EMAIL_APP_PASSWORD environment variables are not set.")
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
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def format_github_markdown(internships_df, jobs_df, stats):
    """Generates Markdown formatting for a GitHub Issue payload."""
    import pandas as pd
    internships_df = sort_by_match_percentage(internships_df.copy() if not internships_df.empty else pd.DataFrame())
    jobs_df = sort_by_match_percentage(jobs_df.copy() if not jobs_df.empty else pd.DataFrame())

    md = "## Automated AI Job Alerts\n\n"
    md += f"**Pipeline Stats:** Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: {stats['approved']}\n\n---\n\n"
    
    md += "### 🎓 Internships (AI & SWE)\n\n"
    if internships_df.empty:
        md += "No relevant internships found today.\n\n"
    else:
        md += "| Title | Company | Location | AI Verdict | Match % | Link |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, row in internships_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if str(match_pct) != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            md += f"| {title} | {company} | {location} | {verdict} | {match_pct} | [Apply]({job_url}) |\n"
        md += "\n"
        
    md += "### 💼 Junior & Entry-Level Jobs (CV-Matched)\n\n"
    if jobs_df.empty:
        md += "No relevant full-time jobs found today.\n\n"
    else:
        md += "| Title | Company | Location | AI Verdict | Match % | Link |\n"
        md += "|---|---|---|---|---|---|\n"
        for _, row in jobs_df.iterrows():
            title = row.get("title", "N/A")
            company = row.get("company", "N/A")
            location = row.get("location", "Remote/Unspecified")
            verdict = row.get("ai_verdict", "")
            match_pct = row.get("match_percentage", "N/A")
            if str(match_pct) != "N/A": match_pct = f"{match_pct}%"
            job_url = row.get("job_url", "#")
            md += f"| {title} | {company} | {location} | {verdict} | {match_pct} | [Apply]({job_url}) |\n"
            
    return md

def create_github_issue(title, body):
    """Posts a new GitHub Issue with the collected jobs."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    
    if not token or not repo:
        print("Error: GITHUB_TOKEN or GITHUB_REPOSITORY environment variables are not set.")
        return
        
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": title,
        "body": body
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"GitHub Issue created successfully: {response.json().get('html_url')}")
    except Exception as e:
        print(f"Failed to create GitHub Issue: {e}")

def cleanup_old_github_issues(days_old=5):
    """Closes 'Automated AI Job Alerts' issues that are older than `days_old` days to keep the repo clean."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    
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
        
        now = datetime.utcnow()
        for issue in issues:
            if "Automated AI Job Alerts" in issue.get("title", ""):
                created_at_str = issue.get("created_at")
                if created_at_str:
                    created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
                    age_days = (now - created_at).days
                    if age_days >= days_old:
                        issue_num = issue.get("number")
                        print(f"Closing old issue #{issue_num} (Age: {age_days} days)")
                        patch_url = f"https://api.github.com/repos/{repo}/issues/{issue_num}"
                        requests.patch(patch_url, headers=headers, json={"state": "closed"})
                        
    except Exception as e:
        print(f"Failed to cleanup old GitHub issues: {e}")
