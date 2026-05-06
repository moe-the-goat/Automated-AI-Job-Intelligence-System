import json
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jobspy import scrape_jobs
import pandas as pd
import requests
from datetime import datetime
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def format_email_html(all_jobs_df):
    if all_jobs_df.empty:
        return "<p>No new jobs found matching your criteria.</p>"
    
    html = "<h2>New Job Postings</h2>"
    html += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
    html += "<tr><th>Title</th><th>Company</th><th>Location</th><th>Link</th></tr>"
    
    for _, row in all_jobs_df.iterrows():
        title = row.get("title", "N/A")
        company = row.get("company", "N/A")
        
        # Safely handle location fields
        loc = row.get("location", "")
        if pd.notna(loc) and loc:
            location = str(loc)
        else:
            city = row.get("city", "") if pd.notna(row.get("city")) else ""
            state = row.get("state", "") if pd.notna(row.get("state")) else ""
            country = row.get("country", "") if pd.notna(row.get("country")) else ""
            parts = [p for p in [city, state, country] if p]
            location = ", ".join(parts)
            
        if not location:
            location = "Remote/Unspecified"
            
        job_url = row.get("job_url", "#")
        
        html += f"<tr><td>{title}</td><td>{company}</td><td>{location}</td><td><a href='{job_url}'>Apply</a></td></tr>"
    
    html += "</table>"
    return html

def fetch_remotive_jobs():
    try:
        url = "https://remotive.com/api/remote-jobs?category=software-dev"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
        
        parsed_jobs = []
        for j in jobs:
            parsed_jobs.append({
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("candidate_required_location", "Remote"),
                "job_url": j.get("url", ""),
                "date_posted": j.get("publication_date", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Remotive jobs: {e}")
        return pd.DataFrame()

def fetch_arbeitnow_jobs():
    try:
        url = "https://www.arbeitnow.com/api/job-board-api"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("data", [])
        
        parsed_jobs = []
        for j in jobs:
            if j.get("remote"):
                parsed_jobs.append({
                    "title": j.get("title", ""),
                    "company": j.get("company_name", ""),
                    "location": "Remote",
                    "job_url": j.get("url", ""),
                    "date_posted": str(j.get("created_at", ""))
                })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Arbeitnow jobs: {e}")
        return pd.DataFrame()

def filter_api_jobs(df, hours_old):
    if df.empty:
        return df
    
    # 1. Filter by role and experience keywords
    role_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'machine learning', 'backend', 'frontend', 'fullstack']
    exp_keywords = ['junior', 'jr', 'entry', 'intern', 'internship', 'graduate', 'grad']
    
    title_lower = df['title'].str.lower()
    has_role = title_lower.str.contains('|'.join(role_keywords), na=False)
    has_exp = title_lower.str.contains('|'.join([rf'\b{w}\b' for w in exp_keywords]), na=False)
    df = df[has_role & has_exp].copy()
    
    # 2. Filter by recency (hours_old)
    try:
        df['date_posted_dt'] = pd.to_datetime(df['date_posted'], utc=True, errors='coerce')
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(hours=hours_old)
        df = df[df['date_posted_dt'].isna() | (df['date_posted_dt'] >= cutoff)]
        df = df.drop(columns=['date_posted_dt'])
    except Exception as e:
        print(f"Date filtering error: {e}")
        
    return df

def send_email(subject, html_content, email_settings):
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
        # Connect to SMTP server (using SSL for port 465)
        with smtplib.SMTP_SSL(email_settings["smtp_server"], email_settings["smtp_port"]) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def format_github_markdown(all_jobs_df):
    if all_jobs_df.empty:
        return "No new jobs found matching your criteria."
    
    md = "## New Job Postings\n\n"
    md += "| Title | Company | Location | Link |\n"
    md += "|---|---|---|---|\n"
    
    for _, row in all_jobs_df.iterrows():
        title = row.get("title", "N/A")
        company = row.get("company", "N/A")
        
        # Safely handle location fields
        loc = row.get("location", "")
        if pd.notna(loc) and loc:
            location = str(loc)
        else:
            city = row.get("city", "") if pd.notna(row.get("city")) else ""
            state = row.get("state", "") if pd.notna(row.get("state")) else ""
            country = row.get("country", "") if pd.notna(row.get("country")) else ""
            parts = [p for p in [city, state, country] if p]
            location = ", ".join(parts)
            
        if not location:
            location = "Remote/Unspecified"
            
        job_url = row.get("job_url", "#")
        
        md += f"| {title} | {company} | {location} | [Apply]({job_url}) |\n"
    
    return md

def create_github_issue(title, body):
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

def main():
    config = load_config()
    all_jobs_dfs = []
    
    print("Starting job scrape...")
    for search in config.get("searches", []):
        print(f"Scraping for: {search.get('search_term')} in {search.get('location')}...")
        try:
            jobs = scrape_jobs(
                site_name=search.get("site_name", ["linkedin", "indeed", "glassdoor"]),
                search_term=search.get("search_term"),
                location=search.get("location"),
                distance=search.get("distance", 50),
                job_type=search.get("job_type"),
                is_remote=search.get("is_remote", False),
                results_wanted=search.get("results_wanted", 20),
                hours_old=search.get("hours_old", 24),
                country_indeed=search.get("country_indeed", "USA")
            )
            print(f"Found {len(jobs)} jobs for this search.")
            all_jobs_dfs.append(jobs)
        except Exception as e:
            print(f"Error scraping for {search.get('search_term')}: {e}")
            
    # Determine the maximum hours_old from config to use for APIs
    max_hours = 24
    if config.get("searches"):
        max_hours = max([s.get("hours_old", 24) for s in config.get("searches", [])])

    # Fetch from secondary APIs
    print("Fetching from Remotive API...")
    remotive_df = fetch_remotive_jobs()
    if not remotive_df.empty:
        remotive_df = filter_api_jobs(remotive_df, hours_old=max_hours)
        print(f"Found {len(remotive_df)} relevant jobs from Remotive.")
        if not remotive_df.empty:
            all_jobs_dfs.append(remotive_df)
            
    print("Fetching from Arbeitnow API...")
    arbeitnow_df = fetch_arbeitnow_jobs()
    if not arbeitnow_df.empty:
        arbeitnow_df = filter_api_jobs(arbeitnow_df, hours_old=max_hours)
        print(f"Found {len(arbeitnow_df)} relevant jobs from Arbeitnow.")
        if not arbeitnow_df.empty:
            all_jobs_dfs.append(arbeitnow_df)
            
    if all_jobs_dfs:
        combined_jobs = pd.concat(all_jobs_dfs, ignore_index=True)
        
        # Drop duplicates by URL
        if "job_url" in combined_jobs.columns:
            combined_jobs = combined_jobs.drop_duplicates(subset=["job_url"])
            
        # Drop duplicates by exact Title + Company
        if "title" in combined_jobs.columns and "company" in combined_jobs.columns:
            combined_jobs = combined_jobs.drop_duplicates(subset=["title", "company"])
            
        # Filter out senior/lead roles
        exclude_words = ['senior', 'sr', 'sr.', 'lead', 'principal', 'manager', 'director', 'staff', 'head', 'vp', 'president']
        if "title" in combined_jobs.columns:
            pattern = '|'.join([rf'\b{w}\b' for w in exclude_words])
            combined_jobs = combined_jobs[~combined_jobs['title'].str.lower().str.contains(pattern, na=False)]
            
        # Ensure job title is highly relevant (must contain at least one tech keyword)
        role_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'machine learning', 'backend', 'frontend', 'fullstack', 'web', 'python', 'java', 'c\\+\\+', 'c#', 'programmer']
        if "title" in combined_jobs.columns:
            pattern = '|'.join([rf'{w}' for w in role_keywords])
            combined_jobs = combined_jobs[combined_jobs['title'].str.lower().str.contains(pattern, na=False)]
            
        print(f"Total unique jobs found: {len(combined_jobs)}")
        
        if not combined_jobs.empty:
            output_config = config.get("output", {"use_email": True, "use_github_issue": False})
            
            if output_config.get("use_email"):
                html_content = format_email_html(combined_jobs)
                send_email("Your Automated Job Alerts", html_content, config.get("email_settings", {}))
                
            if output_config.get("use_github_issue"):
                md_content = format_github_markdown(combined_jobs)
                today = datetime.now().strftime("%Y-%m-%d")
                create_github_issue(f"Automated Job Alerts - {today}", md_content)
        else:
            print("No new jobs found.")
    else:
        print("No job data collected.")

if __name__ == "__main__":
    main()
