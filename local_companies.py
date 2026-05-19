import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import json
import re
import time
from urllib.parse import urlparse
# Heavy deps (`ddgs`, `jobspy`) are lazy-imported inside the functions that
# actually need them so the pure helpers (linkedin_post_date, handle matcher)
# can be unit-tested without those packages installed.

from pipeline.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

from pipeline.core_filter import apply_pipeline_filters, JobTracker
from pipeline.core_ai import evaluate_job_with_ai, quick_viability_check, skipped_result
from pipeline.core_ats import (
    extract_linkedin_handle,
    get_jobs_for_company as get_ats_jobs,
    AtsCache,
)
from pipeline.url_validation import is_job_url_like, probe_urls_alive_batch
from pipeline.core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues

# How far back the local pipeline will accept LinkedIn posts (matches the JobSpy 6-day window).
LOCAL_LOOKBACK_DAYS = 6

# LinkedIn activity IDs are snowflake-like: the top ~41 bits encode a UNIX timestamp
# in MILLISECONDS (not seconds!). Right-shifting the 64-bit ID by 22 strips off the
# low sequence/counter bits and yields a millisecond timestamp.
# Verified on real URLs: 7397959444342575104 -> 1763830509824 ms -> 2025-11-22 UTC.
_LINKEDIN_ACTIVITY_RE = re.compile(r'activity-(\d+)')
_LINKEDIN_HANDLE_RE = re.compile(r'linkedin\.com/posts/([a-z0-9\-]+?)(?:_|/)')

def linkedin_post_date(url):
    """Decode the post date from a LinkedIn activity URL. Returns None if undecodable.

    BUG HISTORY: an earlier version treated the shifted value as seconds, which
    overflowed fromtimestamp() and made the function always return None. The
    filter that used it then never fired and 5-month-old + 1-year-old posts
    slipped into the email. Fixed by dividing by 1000 (ms -> s).
    """
    match = _LINKEDIN_ACTIVITY_RE.search(url or "")
    if not match:
        return None
    try:
        activity_id = int(match.group(1))
        ts_milliseconds = activity_id >> 22
        return datetime.fromtimestamp(ts_milliseconds / 1000, tz=timezone.utc)
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

def ddg_search_for_jobs(company_name, domain, linkedin_handle=None):
    """Uses DuckDuckGo to search for recent job posts on LinkedIn and the company website.

    When `linkedin_handle` is provided (extracted from the Excel-sheet LinkedIn URL),
    the LinkedIn search becomes a precise site:linkedin.com/company/{handle}/posts
    query — far less noise than the old first-word-of-company-name approach.
    """
    from ddgs import DDGS  # lazy import — see top-of-file comment
    jobs_found = []
    ddgs = DDGS()

    short_name = company_name.split()[0] if len(company_name.split()) > 0 else company_name

    # 1. Search LinkedIn Posts (handle-precise when available, fall back to name search)
    try:
        if linkedin_handle:
            q1 = f'site:linkedin.com/company/{linkedin_handle}/posts (hiring OR vacancy OR "looking for" OR job)'
            logger.info("Searching LinkedIn Posts (handle '%s') for: %s...", linkedin_handle, company_name)
        else:
            q1 = f'site:linkedin.com/posts {short_name} (hiring OR vacancy OR "looking for" OR job)'
            logger.info("Searching LinkedIn Posts for: %s (using '%s')...", company_name, short_name)
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
                logger.info("Skipping old post for %s: posted %s", company_name, post_date.date())
                continue
            # Drop unrelated companies that match only because the first word is generic
            # (e.g. 'Future' matching 'pankh-workforce-solution' on a post mentioning FIS).
            if not linkedin_handle_matches(link, company_name):
                logger.info("Skipping unrelated post for %s: handle mismatch (%s)", company_name, link[:80])
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
        logger.warning("DDG LinkedIn search failed for %s: %s", company_name, e)

    # 2. Search Company Website
    if domain:
        try:
            q2 = f'site:{domain} (hiring OR careers OR jobs OR vacancy)'
            logger.info("Searching Website (%s) for: %s...", domain, company_name)
            res2 = ddgs.text(q2, max_results=3, timelimit="w") # past week
            for r in res2:
                title = r.get('title', '')
                body = r.get('body', '')
                link = r.get('href', '')

                # URL-pattern check: a result for `site:freightos.com (hiring
                # OR careers OR jobs OR vacancy)` once matched the URL
                # /freight-industry-updates/market-updates/the-data-behind-
                # amazons-logistics-and-fulfillment-play/ — a blog post about
                # logistics, not a job. is_job_url_like requires the path to
                # look like an actual job-posting page.
                if not is_job_url_like(link):
                    logger.info("Skipping non-job URL for %s: %s", company_name, link[:80])
                    continue

                jobs_found.append({
                    "title": "Website Job: " + title[:50] + "...",
                    "company": company_name,
                    "location": "Local/Remote",
                    "job_url": link,
                    "description": body,
                    "job_type": "fulltime"
                })
        except Exception as e:
            logger.warning("DDG Website search failed for %s: %s", company_name, e)
            
    time.sleep(1) # Be nice to DDG API
    return jobs_found

def load_local_companies():
    """Loads all companies from the Excel files.

    The two sheets use slightly different column casing (`LinkedIn Profile` vs
    `LinkedIn profile`, `Jobs Website` vs `Jobs website`). We normalize to lower-case
    field names so the rest of the pipeline can read them uniformly.
    """
    files = ["IT Companies - Nablus.xlsx", "IT Companies - Ramallah.xlsx"]
    dfs = []
    for f in files:
        if os.path.exists(f):
            try:
                df = pd.read_excel(f)
                # Normalize column names: trim + lowercase.
                df.columns = [str(c).strip().lower() for c in df.columns]
                dfs.append(df)
            except Exception as e:
                logger.warning("Error loading %s: %s", f, e)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def main():
    configure_logging()
    tracker = JobTracker()
    try:
        run_local_pipeline(tracker)
    finally:
        # Always persist tracker + sweep stale issues, even if the pipeline crashed.
        tracker.save()
        # Sweep BOTH the private logs repo (PAT) and the legacy public repo
        # (default token) so historical issues there also fade out in 2 days.
        logs_repo = os.environ.get("LOGS_REPO")
        logs_token = os.environ.get("LOGS_REPO_TOKEN")
        logger.info("Running GitHub Issue cleanup...")
        if logs_repo and logs_token:
            cleanup_old_github_issues(days_old=2, repo=logs_repo, token=logs_token)
        cleanup_old_github_issues(days_old=2)

def run_local_pipeline(tracker):
    logger.info("Starting Local Companies Scrape...")
    companies_df = load_local_companies()
    if companies_df.empty:
        logger.warning("No companies loaded. Ensure the Excel files exist.")
        return
        
    all_raw_jobs = []
    ats_cache = AtsCache()
    # Load API keys early. Gemini stays for the Jina-fallback branch in the ATS
    # sweep. The main AI verdict now runs on Cerebras (primary) + Groq (fallback).
    # local_companies.py does NOT run Layer 3 geo-checks — Palestinian companies
    # don't have non-Palestine geo-restriction concerns.
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    # Track statistics
    stats = {"scraped": 0, "filtered": 0, "approved": 0, "ats_jobs": 0, "jina_jobs": 0}

    # 1. Scrape Jobs for each company
    for _, row in companies_df.iterrows():
        company_name = str(row.get("company name", "")).strip()
        website = str(row.get("jobs website", ""))
        linkedin_url = str(row.get("linkedin profile", ""))
        domain = extract_domain(website)
        linkedin_handle = extract_linkedin_handle(linkedin_url) if linkedin_url and linkedin_url.lower() != "nan" else None

        if not company_name or company_name == "nan":
            continue

        # NEW: ATS API scrape (Greenhouse / Lever / Workable / Ashby / Workday).
        # One-time detection cached in data/ats_cache.json so subsequent runs go
        # straight to the API. `website` doubles as the careers-page seed for
        # first-time detection.
        #
        # Wave 2: when no SaaS ATS is detected, route through Jina Reader +
        # Gemini extraction. This costs one extra Gemini call per ATS-less
        # company per run but unlocks the long tail of custom careers pages
        # (most Palestinian companies fall here).
        if website and website.lower() != "nan":
            try:
                ats_jobs = get_ats_jobs(
                    company_name, website, cache=ats_cache,
                    gemini_api_key=gemini_key,
                    jina_fallback=bool(gemini_key),
                )
                if ats_jobs:
                    logger.info("ATS yielded %d job(s) for %s", len(ats_jobs), company_name)
                    stats["ats_jobs"] += len(ats_jobs)
                    all_raw_jobs.extend(ats_jobs)
            except Exception as e:
                logger.warning("ATS scrape failed for %s: %s", company_name, str(e)[:120])

        # DuckDuckGo Scrapes (now precision-boosted with linkedin_handle when available)
        ddg_jobs = ddg_search_for_jobs(company_name, domain, linkedin_handle=linkedin_handle)
        all_raw_jobs.extend(ddg_jobs)

        # JobSpy Scrape (Jobs Section)
        try:
            from jobspy import scrape_jobs  # lazy import
            logger.info("Running JobSpy for %s...", company_name)
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
            logger.warning("JobSpy failed for %s: %s", company_name, e)

    # Persist ATS cache for future runs (so re-detection is rare).
    ats_cache.save()
    if stats["ats_jobs"]:
        logger.info("ATS sweep contributed %d job(s) this run.", stats['ats_jobs'])

    # Ghost-listing check: DDG/Bing index stale URLs for weeks after a company
    # removes a job. Batch HEAD-probe every DDG-sourced URL and drop the dead
    # ones BEFORE the AI ever sees them. We skip URLs from the ATS API path
    # (those came from live endpoints, no need to verify) and JobSpy (which
    # already filters by hours_old). The probe is concurrent so 30 URLs take
    # ~1.5s instead of 30s.
    ddg_urls = [
        j.get("job_url") for j in all_raw_jobs
        if str(j.get("title", "")).startswith(("LinkedIn Post:", "Website Job:"))
        and j.get("job_url")
    ]
    if ddg_urls:
        unique_urls = list(set(ddg_urls))
        logger.info("Verifying %d DDG-sourced URL(s) via HEAD probe...", len(unique_urls))
        alive_map = probe_urls_alive_batch(unique_urls)
        before = len(all_raw_jobs)
        all_raw_jobs = [
            j for j in all_raw_jobs
            if not str(j.get("title", "")).startswith(("LinkedIn Post:", "Website Job:"))
            or alive_map.get(j.get("job_url"), True)        # default True so unprobed entries stay
        ]
        dropped = before - len(all_raw_jobs)
        if dropped:
            logger.info("Ghost-listing filter: dropped %d dead URL(s).", dropped)

    if not all_raw_jobs:
        logger.info("No jobs found at all. Shutting down quietly.")
        return
        
    combined_jobs = pd.DataFrame(all_raw_jobs)
    stats['scraped'] = len(combined_jobs)
    logger.info("Total raw jobs found: %d", stats['scraped'])
    
    # 2. Filter Jobs (tracker drops previously-seen URLs first)
    combined_jobs = apply_pipeline_filters(combined_jobs, tracker=tracker)
    stats['filtered'] = len(combined_jobs)
    logger.info("Total jobs surviving pre-filters: %d", stats['filtered'])
    
    # 3. AI Evaluation (gemini_key was loaded earlier for the Jina fallback)
    try:
        with open("cv_text.txt", "r", encoding="utf-8") as f:
            cv_text = f.read()
    except:
        cv_text = "Computer Engineering student specializing in AI systems engineering, building end-to-end pipelines that integrate LLMs, embeddings, and multi-source data into production-ready backend systems. Experienced deploying Python-based solutions with REST APIs, automated workflows, and real-world constraints. Growing focus on Generative AI, RAG architectures, and scalable intelligent systems."
        
    verdicts, valid_mask, match_pcts = [], [], []
    tech_fits, exp_fits, log_fits = [], [], []
    comps, efforts, suspiciouses, scams = [], [], [], []
    blacklisteds = []
    prescreen_skipped = 0

    if not combined_jobs.empty:
        logger.info("Running AI Job Validation...")
        for idx, row in combined_jobs.iterrows():
            is_viable, reason = quick_viability_check(row)
            if not is_viable:
                prescreen_skipped += 1
                logger.info("[SKIP] %-55s -> %s", str(row.get('title', ''))[:55], reason)
                result = skipped_result(reason)
                evaluated = True
            else:
                result, evaluated = evaluate_job_with_ai(row, cv_text, cerebras_key, groq_key)
            verdicts.append(result["verdict"])
            valid_mask.append(result["is_valid"])
            match_pcts.append(result["match_percentage"])
            tech_fits.append(result["tech_fit"])
            exp_fits.append(result["experience_fit"])
            log_fits.append(result["logistics_fit"])
            comps.append(result["compensation"])
            efforts.append(result["effort"])
            suspiciouses.append(result["suspicious"])
            scams.append(result.get("scam", False))
            blacklisteds.append(bool(row.get("pre_flagged_low_quality", False)))
            # Only mark seen on real verdicts; errors get retried next run.
            if evaluated:
                tracker.mark_seen(str(row.get("job_url", "")))

        logger.info("Pre-screen summary: skipped %d / %d jobs before AI eval.", prescreen_skipped, len(combined_jobs))
        combined_jobs['ai_verdict'] = verdicts
        combined_jobs['match_percentage'] = match_pcts
        combined_jobs['tech_fit'] = tech_fits
        combined_jobs['experience_fit'] = exp_fits
        combined_jobs['logistics_fit'] = log_fits
        combined_jobs['compensation'] = comps
        combined_jobs['effort'] = efforts
        combined_jobs['suspicious'] = suspiciouses
        combined_jobs['scam'] = scams
        combined_jobs['pre_flagged_low_quality'] = blacklisteds

        # Filter down to approved
        approved_jobs = combined_jobs[valid_mask]
        stats['approved'] = len(approved_jobs)
    else:
        approved_jobs = combined_jobs
        stats['approved'] = 0

    logger.info("Total jobs approved by AI: %d", stats['approved'])
    
    # Load config for email settings
    with open("config.json", "r") as f:
        config = json.load(f)
        
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 4. Routing Logic
    logs_repo = os.environ.get("LOGS_REPO")
    logs_token = os.environ.get("LOGS_REPO_TOKEN")
    if stats['approved'] > 0:
        # We have approved jobs. Send Email and create GitHub issue.
        intern_mask = approved_jobs['title'].str.lower().str.contains('intern')
        internships_df = approved_jobs[intern_mask]
        jobs_df = approved_jobs[~intern_mask]

        html_content = format_email_html(internships_df, jobs_df, stats)
        send_email(f"Local Companies Job Alerts - {today}", html_content, config.get("email_settings", {}))

        md_content = format_github_markdown(internships_df, jobs_df, stats)
        create_github_issue(
            f"Local Companies Job Alerts - {today}",
            md_content,
            repo=logs_repo,
            token=logs_token,
        )

    elif stats['scraped'] > 0 and stats['approved'] == 0:
        # We found jobs but none passed. Create GitHub issue ONLY.
        title = f"Local Companies Scan - {today} (0 Passed)"
        body = f"## Local Companies Job Scan\n\n**Pipeline Stats:** Scraped: {stats['scraped']} &rarr; Filtered to: {stats['filtered']} &rarr; AI Approved: 0\n\nNo jobs passed the AI validation today. Did not send an email."
        create_github_issue(title, body, repo=logs_repo, token=logs_token)

if __name__ == "__main__":
    main()
