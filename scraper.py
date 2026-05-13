import os
import json
import pandas as pd
from datetime import datetime
from jobspy import scrape_jobs

from core_search import (
    fetch_remotive_jobs,
    fetch_arbeitnow_jobs,
    fetch_jobicy_jobs,
    fetch_remoteok_jobs
)
from core_filter import filter_api_jobs, apply_pipeline_filters
from core_ai import evaluate_job_with_ai
from core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues

"""
MAIN SCRAPER EXECUTOR
---------------------
This is the master script. It orchestrates the entire pipeline:
1. Triggers the Search module to gather raw jobs.
2. Passes them to the Filter module to drop duplicates and irrelevant roles.
3. Feeds the survivors 1-by-1 to the AI module.
4. Hands the final approved list to the Notify module for dispatch.
"""

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    config = load_config()
    all_jobs_dfs = []
    
    print("Starting job scrape...")
    
    # Track statistics for the daily email header (Tier 3 Item 14)
    stats = {"scraped": 0, "filtered": 0, "approved": 0}
    
    # 1. JobSpy Scrapes (LinkedIn, Glassdoor, Indeed)
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

    # 2. Secondary API Scrapes
    print("Fetching from Remotive API...")
    remotive_df = fetch_remotive_jobs()
    if not remotive_df.empty:
        remotive_df = filter_api_jobs(remotive_df, hours_old=max_hours)
        all_jobs_dfs.append(remotive_df)
            
    print("Fetching from Arbeitnow API...")
    arbeitnow_df = fetch_arbeitnow_jobs()
    if not arbeitnow_df.empty:
        arbeitnow_df = filter_api_jobs(arbeitnow_df, hours_old=max_hours)
        all_jobs_dfs.append(arbeitnow_df)
            
    print("Fetching from Jobicy API...")
    jobicy_df = fetch_jobicy_jobs()
    if not jobicy_df.empty:
        jobicy_df = filter_api_jobs(jobicy_df, hours_old=max_hours)
        all_jobs_dfs.append(jobicy_df)

    print("Fetching from RemoteOK API...")
    remoteok_df = fetch_remoteok_jobs()
    if not remoteok_df.empty:
        remoteok_df = filter_api_jobs(remoteok_df, hours_old=max_hours)
        all_jobs_dfs.append(remoteok_df)
            
    # 3. Compile and Filter
    if all_jobs_dfs:
        combined_jobs = pd.concat(all_jobs_dfs, ignore_index=True)
        stats['scraped'] = len(combined_jobs)
        
        # Apply the gauntlet of deterministic filters
        combined_jobs = apply_pipeline_filters(combined_jobs)
        stats['filtered'] = len(combined_jobs)
        print(f"Total unique, unseen jobs surviving the pre-filters: {stats['filtered']}")
        
        if not combined_jobs.empty:
            # 4. AI Evaluation
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            try:
                with open("cv_text.txt", "r", encoding="utf-8") as f:
                    cv_text = f.read()
            except:
                cv_text = "Computer Engineering student, strong in Python, PyTorch, FastAPI, RAG, ML, and Backend Development."
            
            print("Running AI Job Validation 1-by-1 (this may take a while)...")
            verdicts = []
            valid_mask = []
            match_pcts = []
            
            for idx, row in combined_jobs.iterrows():
                verdict, is_valid, match_pct = evaluate_job_with_ai(row, cv_text, gemini_key)
                verdicts.append(verdict)
                valid_mask.append(is_valid)
                match_pcts.append(match_pct)
                
            combined_jobs['ai_verdict'] = verdicts
            combined_jobs['match_percentage'] = match_pcts
            
            # Filter down to only AI-approved jobs
            combined_jobs = combined_jobs[valid_mask]
            stats['approved'] = len(combined_jobs)
            print(f"Total jobs remaining after AI validation: {stats['approved']}")
            
            # 5. Output and Notification
            # Split into internships and jobs
            intern_mask = combined_jobs['title'].str.lower().str.contains('intern') | (combined_jobs.get('job_type', pd.Series(dtype=str)).astype(str).str.lower().str.contains('internship'))
            internships_df = combined_jobs[intern_mask]
            jobs_df = combined_jobs[~intern_mask]
            
            output_config = config.get("output", {"use_email": True, "use_github_issue": False})
            
            if output_config.get("use_email"):
                html_content = format_email_html(internships_df, jobs_df, stats)
                send_email("Your Automated AI Job Alerts", html_content, config.get("email_settings", {}))
                
            if output_config.get("use_github_issue"):
                md_content = format_github_markdown(internships_df, jobs_df, stats)
                today = datetime.now().strftime("%Y-%m-%d")
                create_github_issue(f"Automated AI Job Alerts - {today}", md_content)
                
                # Clean up old issues to keep the repo clean
                print("Running GitHub Issue cleanup...")
                cleanup_old_github_issues(days_old=5)
        else:
            print("No new jobs survived the filters today.")
    else:
        print("No job data collected from any sources.")

if __name__ == "__main__":
    main()
