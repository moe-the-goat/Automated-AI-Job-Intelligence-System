import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import re
import time
from ddgs import DDGS
from urllib.parse import urlparse
from jobspy import scrape_jobs

from core_filter import apply_pipeline_filters, JobTracker
from core_ai import evaluate_job_with_ai
from core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues

# How far back the local pipeline will accept LinkedIn posts (matches the JobSpy 6-day window).
LOCAL_LOOKBACK_DAYS = 6

# LinkedIn activity IDs are snowflake-like: right-shifting by 22 yields a Unix epoch
# in seconds. Verified on real URLs (7397959444342575104 -> 2025-11-24).
_LINKEDIN_ACTIVITY_RE = re.compile(r'activity-(\d+)')
_LINKEDIN_HANDLE_RE = re.compile(r'linkedin\.com/posts/([a-z0-9\-]+?)(?:_|/)')

def linkedin_post_date(url):
    """Decode the post date from a LinkedIn activity URL. Returns None if undecodable."""
    match = _LINKEDIN_ACTIVITY_RE.search(url or "")
    if not match:
        return None
    try:
        activity_id = int(match.group(1))
        ts_seconds = activity_id >> 22
        return datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None

def linkedin_handle_matches(url, company_name):
    """True if the LinkedIn post URL's handle contains a meaningful chunk of the company name.

    Prevents DDG false positives where a generic first word like 'Future' matches a totally
    unrelated post (e.g. linkedin.com/posts/pankh-workforce-solution_... for 'Future
    Information Systems').
    """
    match = _LINKEDIN_HANDLE_RE.search((url or "").lower())
    if not match:
        return False
    handle = match.group(1)
    name_tokens = re.findall(r'[a-z]+', company_name.lower())
    return any(len(tok) >= 3 and tok in handle for tok in name_tokens)

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
    
    # Relax company name to the first highly identifiable word (e.g., "ASAL Technologies" -> "ASAL")
    # This helps catch posts by "asaltech" or those that just say "ASAL is hiring"
    short_name = company_name.split()[0] if len(company_name.split()) > 0 else company_name
    
    # 1. Search LinkedIn Posts
    try:
        q1 = f'site:linkedin.com/posts {short_name} (hiring OR vacancy OR "looking for" OR job)'
        print(f"Searching LinkedIn Posts for: {company_name} (using '{short_name}')...")
        res1 = ddgs.text(q1, max_results=3, timelimit="w") # past week
        cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
        for r in res1:
            title = r.get('title', '')
            body = r.get('body', '')
            link = r.get('href', '')

            # DDG's `timelimit` is unreliable for LinkedIn — re-verify the post date
            # from the activity ID itself and drop anything older than the lookback window.
            post_date = linkedin_post_date(link)
            if post_date and post_date < cutoff:
                print(f"  Skipping old post for {company_name}: posted {post_date.date()}")
                continue
            # Drop unrelated companies that match only because the first word is generic
            # (e.g. 'Future' matching 'pankh-workforce-solution' on a post mentioning FIS).
            if not linkedin_handle_matches(link, company_name):
                print(f"  Skipping unrelated post for {company_name}: handle mismatch ({link[:80]})")
                continue

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
            q2 = f'site:{domain} (hiring OR careers OR jobs OR vacancy)'
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
    tracker = JobTracker()
    try:
        run_local_pipeline(tracker)
    finally:
        # Always persist tracker + sweep stale issues, even if the pipeline crashed.
        tracker.save()
        print("Running GitHub Issue cleanup...")
        cleanup_old_github_issues(days_old=3)

def run_local_pipeline(tracker):
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
                hours_old=144 # 6 days
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
        return
        
    combined_jobs = pd.DataFrame(all_raw_jobs)
    stats['scraped'] = len(combined_jobs)
    print(f"Total raw jobs found: {stats['scraped']}")
    
    # 2. Filter Jobs (tracker drops previously-seen URLs first)
    combined_jobs = apply_pipeline_filters(combined_jobs, tracker=tracker)
    stats['filtered'] = len(combined_jobs)
    print(f"Total jobs surviving pre-filters: {stats['filtered']}")
    
    # 3. AI Evaluation
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        with open("cv_text.txt", "r", encoding="utf-8") as f:
            cv_text = f.read()
    except:
        cv_text = "Computer Engineering student, strong in Python, PyTorch, FastAPI, Backend."
        
    verdicts, valid_mask, match_pcts = [], [], []
    tech_fits, exp_fits, log_fits = [], [], []
    comps, efforts, suspiciouses = [], [], []
    blacklisteds = []

    if not combined_jobs.empty:
        print("Running AI Job Validation...")
        for idx, row in combined_jobs.iterrows():
            result, evaluated = evaluate_job_with_ai(row, cv_text, gemini_key)
            verdicts.append(result["verdict"])
            valid_mask.append(result["is_valid"])
            match_pcts.append(result["match_percentage"])
            tech_fits.append(result["tech_fit"])
            exp_fits.append(result["experience_fit"])
            log_fits.append(result["logistics_fit"])
            comps.append(result["compensation"])
            efforts.append(result["effort"])
            suspiciouses.append(result["suspicious"])
            blacklisteds.append(bool(row.get("pre_flagged_low_quality", False)))
            # Only mark seen on real verdicts; errors get retried next run.
            if evaluated:
                tracker.mark_seen(str(row.get("job_url", "")))

        combined_jobs['ai_verdict'] = verdicts
        combined_jobs['match_percentage'] = match_pcts
        combined_jobs['tech_fit'] = tech_fits
        combined_jobs['experience_fit'] = exp_fits
        combined_jobs['logistics_fit'] = log_fits
        combined_jobs['compensation'] = comps
        combined_jobs['effort'] = efforts
        combined_jobs['suspicious'] = suspiciouses
        combined_jobs['pre_flagged_low_quality'] = blacklisteds

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

if __name__ == "__main__":
    main()
