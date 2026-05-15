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
from core_filter import filter_api_jobs, apply_pipeline_filters, JobTracker
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
    tracker = JobTracker()
    try:
        run_pipeline(config, tracker)
    finally:
        # Always persist tracker + sweep stale issues, even if the pipeline crashed.
        tracker.save()
        print("Running GitHub Issue cleanup...")
        cleanup_old_github_issues(days_old=3)

# Public APIs don't filter by recency server-side, so we look back further
# than the 24h JobSpy window to surface more remote jobs from non-LinkedIn sources.
API_HOURS_OLD = 72

def run_pipeline(config, tracker):
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
    for name, fetch_fn in [
        ("Remotive", fetch_remotive_jobs),
        ("Arbeitnow", fetch_arbeitnow_jobs),
        ("Jobicy", fetch_jobicy_jobs),
        ("RemoteOK", fetch_remoteok_jobs),
    ]:
        print(f"Fetching from {name} API...")
        raw = fetch_fn()
        if raw.empty:
            print(f"  {name}: returned 0 jobs.")
            continue
        filtered = filter_api_jobs(raw, hours_old=API_HOURS_OLD)
        print(f"  {name}: {len(raw)} raw -> {len(filtered)} after role+recency filter (last {API_HOURS_OLD}h).")
        if not filtered.empty:
            all_jobs_dfs.append(filtered)
            
    # 3. Compile and Filter
    if all_jobs_dfs:
        combined_jobs = pd.concat(all_jobs_dfs, ignore_index=True)
        stats['scraped'] = len(combined_jobs)
        
        # Apply the gauntlet of deterministic filters (tracker drops seen URLs first)
        combined_jobs = apply_pipeline_filters(combined_jobs, tracker=tracker)
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
            verdicts, valid_mask, match_pcts = [], [], []
            tech_fits, exp_fits, log_fits = [], [], []
            comps, efforts, suspiciouses = [], [], []

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
                # Only mark seen when AI actually returned a verdict. Quota/timeout
                # errors leave the job unmarked so we can retry it next run.
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
        else:
            print("No new jobs survived the filters today.")
    else:
        print("No job data collected from any sources.")

if __name__ == "__main__":
    main()
