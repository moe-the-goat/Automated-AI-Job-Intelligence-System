import os
import pandas as pd
from datetime import datetime
import json
import time
from duckduckgo_search import DDGS
from urllib.parse import urlparse
from jobspy import scrape_jobs

from core_filter import apply_pipeline_filters, JobTracker
from core_ai import evaluate_job_with_ai
from core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues

"""
LOCAL COMPANIES SCRAPER
-----------------------
This script runs on a separate schedule (every 2 days) to hunt for jobs from 
specific local Palestinian IT companies. It uses DuckDuckGo to bypass LinkedIn 
login walls and search company posts directly, as well as scraping their custom websites.
"""

def extract_domain(url):
    """Extracts the base domain from a URL (e.g. https://www.company.com/jobs -> company.com)"""
    if pd.isna(url) or not str(url).strip():
        return ""
    try:
        domain = urlparse(str(url)).netloc
        domain = domain.replace('www.', '')
        return domain
    except:
        return ""

def ddg_search_for_jobs(company_name, domain):
    """Uses DuckDuckGo to search for recent job posts on LinkedIn and the company website."""
    jobs_found = []
    ddgs = DDGS()
    
    # 1. Search LinkedIn Posts
    try:
        q1 = f'site:linkedin.com/posts "{company_name}" (hiring OR vacancy OR "looking for" OR job)'
        print(f"Searching LinkedIn Posts for: {company_name}...")
        res1 = ddgs.text(q1, max_results=3, timelimit="w") # past week
        for r in res1:
            title = r.get('title', '')
            body = r.get('body', '')
            link = r.get('href', '')
            
            # Simple heuristic to check if it's a tech job
            tech_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'backend', 'frontend']
            if any(k in title.lower() or k in body.lower() for k in tech_keywords):
                jobs_found.append({
                    "title": "LinkedIn Post: " + title[:50] + "...",
                    "company": company_name,
                    "location": "Local/Remote",
                    "job_url": link,
                    "description": body,
                    "job_type": "fulltime"
                })
    except Exception as e:
        print(f"DDG LinkedIn search failed for {company_name}: {e}")

    # 2. Search Company Website
    if domain:
        try:
            q2 = f'site:{domain} (hiring OR careers OR jobs) (software OR developer OR engineer OR ai OR data)'
            print(f"Searching Website ({domain}) for: {company_name}...")
            res2 = ddgs.text(q2, max_results=3, timelimit="w") # past week
            for r in res2:
                title = r.get('title', '')
                body = r.get('body', '')
                link = r.get('href', '')
                
                jobs_found.append({
                    "title": "Website Job: " + title[:50] + "...",
                    "company": company_name,
                    "location": "Local/Remote",
                    "job_url": link,
                    "description": body,
                    "job_type": "fulltime"
                })
        except Exception as e:
            print(f"DDG Website search failed for {company_name}: {e}")
            
    time.sleep(1) # Be nice to DDG API
    return jobs_found

def load_local_companies():
    """Loads all companies from the Excel files."""
    files = ["IT Companies - Nablus.xlsx", "IT Companies - Ramallah.xlsx"]
    dfs = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_excel(f)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def main():
    print("Starting Local Companies Scrape...")
    companies_df = load_local_companies()
    if companies_df.empty:
        print("No companies loaded. Ensure the Excel files exist.")
        return
        
    all_raw_jobs = []
    
    # Track statistics
    stats = {"scraped": 0, "filtered": 0, "approved": 0}
    
    # 1. Scrape Jobs for each company
    for _, row in companies_df.iterrows():
        company_name = str(row.get("Company Name", "")).strip()
        website = str(row.get("Jobs Website", ""))
        domain = extract_domain(website)
        
        if not company_name or company_name == "nan":
            continue
            
        # DuckDuckGo Scrapes
        ddg_jobs = ddg_search_for_jobs(company_name, domain)
        all_raw_jobs.extend(ddg_jobs)
        
        # JobSpy Scrape (Jobs Section)
        try:
            print(f"Running JobSpy for {company_name}...")
            jobspy_res = scrape_jobs(
                site_name=["linkedin"],
                search_term=company_name,
                location="State of Palestine",
                distance=100,
                results_wanted=5,
                hours_old=120 # 5 days
            )
            for _, j_row in jobspy_res.iterrows():
                # Make sure the company name roughly matches to avoid generic search results
                found_company = str(j_row.get("company", "")).lower()
                if company_name.lower() in found_company or found_company in company_name.lower():
                    all_raw_jobs.append(j_row.to_dict())
        except Exception as e:
            print(f"JobSpy failed for {company_name}: {e}")
            
    if not all_raw_jobs:
        print("No jobs found at all. Shutting down quietly.")
        # Cleanup old issues just in case
        cleanup_old_github_issues(days_old=5)
        return
        
    combined_jobs = pd.DataFrame(all_raw_jobs)
    stats['scraped'] = len(combined_jobs)
    print(f"Total raw jobs found: {stats['scraped']}")
    
    # 2. Filter Jobs
    tracker = JobTracker()
    if "job_url" in combined_jobs.columns:
        combined_jobs = combined_jobs[~combined_jobs['job_url'].apply(tracker.is_seen)]
        
    combined_jobs = apply_pipeline_filters(combined_jobs)
    stats['filtered'] = len(combined_jobs)
    print(f"Total jobs surviving pre-filters: {stats['filtered']}")
    
    # 3. AI Evaluation
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        with open("cv_text.txt", "r", encoding="utf-8") as f:
            cv_text = f.read()
    except:
        cv_text = "Computer Engineering student, strong in Python, PyTorch, FastAPI, Backend."
        
    verdicts = []
    valid_mask = []
    match_pcts = []
    
    if not combined_jobs.empty:
        print("Running AI Job Validation...")
        for idx, row in combined_jobs.iterrows():
            verdict, is_valid, match_pct = evaluate_job_with_ai(row, cv_text, gemini_key)
            verdicts.append(verdict)
            valid_mask.append(is_valid)
            match_pcts.append(match_pct)
            
            tracker.mark_seen(str(row.get("job_url", "")))
            
        tracker.save()
        combined_jobs['ai_verdict'] = verdicts
        combined_jobs['match_percentage'] = match_pcts
        
        # Filter down to approved
        approved_jobs = combined_jobs[valid_mask]
        stats['approved'] = len(approved_jobs)
    else:
        approved_jobs = combined_jobs
        stats['approved'] = 0

    print(f"Total jobs approved by AI: {stats['approved']}")
    
    # Load config for email settings
    with open("config.json", "r") as f:
        config = json.load(f)
        
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 4. Routing Logic
    if stats['approved'] > 0:
        # We have approved jobs. Send Email and create GitHub issue.
        intern_mask = approved_jobs['title'].str.lower().str.contains('intern')
        internships_df = approved_jobs[intern_mask]
        jobs_df = approved_jobs[~intern_mask]
        
        html_content = format_email_html(internships_df, jobs_df, stats)
        send_email(f"Local Companies Job Alerts - {today}", html_content, config.get("email_settings", {}))
        
        md_content = format_github_markdown(internships_df, jobs_df, stats)
        create_github_issue(f"Local Companies Job Alerts - {today}", md_content)
        
    elif stats['scraped'] > 0 and stats['approved'] == 0:
        # We found jobs but none passed. Create GitHub issue ONLY.
        title = f"Local Companies Scan - {today} (0 Passed)"
        body = f"## Local Companies Job Scan\n\n**Pipeline Stats:** Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: 0\n\nNo jobs passed the AI validation today. Did not send an email."
        create_github_issue(title, body)
        
    # Finally, clean up old issues
    print("Running GitHub Issue cleanup...")
    cleanup_old_github_issues(days_old=5)

if __name__ == "__main__":
    main()
