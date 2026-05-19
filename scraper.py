import os
import json
import pandas as pd
from datetime import datetime
from jobspy import scrape_jobs

from pipeline.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

from pipeline.core_search import (
    fetch_remotive_jobs,
    fetch_arbeitnow_jobs,
    fetch_jobicy_jobs,
    fetch_remoteok_jobs,
    fetch_himalayas_jobs,
    fetch_themuse_jobs,
    fetch_wwr_jobs,
    fetch_yc_workatastartup_jobs,
)
from pipeline.core_filter import filter_api_jobs, apply_pipeline_filters, JobTracker
from pipeline.core_ai import evaluate_job_with_ai, quick_viability_check, skipped_result
from pipeline.core_geo import should_geo_check, check_remote_eligibility, apply_geo_check_result
from pipeline.core_embedding import attach_similarity
from pipeline.core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues

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
    configure_logging()
    config = load_config()
    tracker = JobTracker()
    try:
        run_pipeline(config, tracker)
    finally:
        # Always persist tracker + sweep stale issues, even if the pipeline crashed.
        tracker.save()
        # New issues now land in the private LOGS_REPO via the LOGS_REPO_TOKEN PAT.
        # We still sweep the legacy public repo (default GITHUB_TOKEN) so any
        # pre-existing issues there fade out within 2 days.
        logs_repo = os.environ.get("LOGS_REPO")
        logs_token = os.environ.get("LOGS_REPO_TOKEN")
        logger.info("Running GitHub Issue cleanup...")
        if logs_repo and logs_token:
            cleanup_old_github_issues(days_old=2, repo=logs_repo, token=logs_token)
        cleanup_old_github_issues(days_old=2)

# Public APIs don't filter by recency server-side, so we look back further
# than the 24h JobSpy window to surface more remote jobs from non-LinkedIn sources.
API_HOURS_OLD = 72

# A3: only the top N jobs (by weighted CV-embedding similarity) get the AI verdict.
# Ranking multiplies raw similarity by region+trust weights so EU / Americas /
# Middle East and trusted big-name companies sort above India-only postings at
# equal raw similarity. A wildcard sample is also evaluated so an imperfectly-
# tuned weighting doesn't permanently hide a long-tail match.
# Top_N progression: 30 -> 45 -> 55 (2026-05-19, after moving main verdict to
# Cerebras+Groq freed up Gemini's quota for Layer 3 geo-checking; we can now
# afford to give the AI eval pass a wider field).
AI_EVAL_TOP_N = 55
WILDCARD_COUNT = 5

def run_pipeline(config, tracker):
    all_jobs_dfs = []

    logger.info("Starting job scrape...")
    
    # Track statistics for the daily email header (Tier 3 Item 14)
    stats = {"scraped": 0, "filtered": 0, "approved": 0}
    
    # 1. JobSpy Scrapes (LinkedIn, Glassdoor, Indeed)
    for search in config.get("searches", []):
        logger.info("Scraping for: %s in %s...", search.get('search_term'), search.get('location'))
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
            logger.info("Found %d jobs for this search.", len(jobs))
            all_jobs_dfs.append(jobs)
        except Exception as e:
            logger.warning("Error scraping for %s: %s", search.get('search_term'), e)
            
    # Determine the maximum hours_old from config to use for APIs
    max_hours = 24
    if config.get("searches"):
        max_hours = max([s.get("hours_old", 24) for s in config.get("searches", [])])

    # 2. Secondary API Scrapes. YC Work at a Startup uses Jina+Gemini extraction
    # so it doesn't take a `hours_old` filter — its results are inherently fresh
    # (the site only lists currently-open roles). We bypass filter_api_jobs for it.
    for name, fetch_fn in [
        ("Remotive", fetch_remotive_jobs),
        ("Arbeitnow", fetch_arbeitnow_jobs),
        ("Jobicy", fetch_jobicy_jobs),
        ("RemoteOK", fetch_remoteok_jobs),
        ("Himalayas", fetch_himalayas_jobs),
        ("TheMuse", fetch_themuse_jobs),
        ("WWR", fetch_wwr_jobs),
    ]:
        logger.info("Fetching from %s API...", name)
        raw = fetch_fn()
        if raw.empty:
            logger.info("%s: returned 0 jobs.", name)
            continue
        filtered = filter_api_jobs(raw, hours_old=API_HOURS_OLD)
        logger.info("%s: %d raw -> %d after role+recency filter (last %dh).", name, len(raw), len(filtered), API_HOURS_OLD)
        if not filtered.empty:
            all_jobs_dfs.append(filtered)

    # YC Work at a Startup — needs Gemini key for the tiered extraction step.
    # If the key isn't set or extraction fails, the fetcher returns an empty
    # DataFrame and the rest of the pipeline continues normally.
    yc_key = os.environ.get("GEMINI_API_KEY", "")
    if yc_key:
        logger.info("Fetching from YC Work at a Startup...")
        yc_raw = fetch_yc_workatastartup_jobs(gemini_api_key=yc_key)
        if not yc_raw.empty:
            # No recency filter — YC's site only lists currently-open roles.
            logger.info("YC Work at a Startup: %d jobs.", len(yc_raw))
            all_jobs_dfs.append(yc_raw)
        else:
            logger.info("YC Work at a Startup: returned 0 jobs.")
            
    # 3. Compile and Filter
    if all_jobs_dfs:
        combined_jobs = pd.concat(all_jobs_dfs, ignore_index=True)
        stats['scraped'] = len(combined_jobs)
        
        # Apply the gauntlet of deterministic filters (tracker drops seen URLs first)
        combined_jobs = apply_pipeline_filters(combined_jobs, tracker=tracker)
        stats['filtered'] = len(combined_jobs)
        logger.info("Total unique, unseen jobs surviving the pre-filters: %d", stats['filtered'])
        
        if not combined_jobs.empty:
            # 4. AI Evaluation
            #  - Gemini key: used by embedding pre-rank AND by core_geo Layer 3 geo-check.
            #  - Cerebras/Groq keys: used by core_ai.evaluate_job_with_ai for the main verdict.
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
            groq_key = os.environ.get("GROQ_API_KEY", "")
            try:
                with open("cv_text.txt", "r", encoding="utf-8") as f:
                    cv_text = f.read()
            except:
                cv_text = "Computer Engineering student specializing in AI systems engineering, building end-to-end pipelines that integrate LLMs, embeddings, and multi-source data into production-ready backend systems. Experienced deploying Python-based solutions with REST APIs, automated workflows, and real-world constraints. Growing focus on Generative AI, RAG architectures, and scalable intelligent systems."

            # A3: pre-rank by CV-embedding similarity. Top-N + wildcards go to AI;
            # the rest still appear in the email under a "Lower-ranked" section.
            combined_jobs = attach_similarity(combined_jobs, cv_text, gemini_key)
            if len(combined_jobs) > AI_EVAL_TOP_N:
                top_n = combined_jobs.head(AI_EVAL_TOP_N)
                rest = combined_jobs.iloc[AI_EVAL_TOP_N:]
                n_wild = min(WILDCARD_COUNT, len(rest))
                wildcards = rest.sample(n_wild, random_state=42) if n_wild > 0 else rest.iloc[:0]
                ai_eval_set = pd.concat([top_n, wildcards]).reset_index(drop=True)
                lower_ranked = rest.drop(wildcards.index).reset_index(drop=True)
                logger.info(
                    "Embedding pre-rank: %d jobs to AI (top %d + %d wildcards); %d lower-ranked deferred.",
                    len(ai_eval_set), AI_EVAL_TOP_N, n_wild, len(lower_ranked),
                )
            else:
                ai_eval_set = combined_jobs
                lower_ranked = pd.DataFrame()
                logger.info("Embedding pre-rank: %d jobs (under threshold, evaluating all).", len(ai_eval_set))
            combined_jobs = ai_eval_set

            logger.info("Running AI Job Validation 1-by-1 (this may take a while)...")
            verdicts, valid_mask, match_pcts = [], [], []
            tech_fits, exp_fits, log_fits = [], [], []
            comps, efforts, suspiciouses, scams = [], [], [], []
            blacklisteds = []
            prescreen_skipped = 0
            geo_checked = 0

            for idx, row in combined_jobs.iterrows():
                is_viable, reason = quick_viability_check(row)
                if not is_viable:
                    prescreen_skipped += 1
                    logger.info("[SKIP] %-55s -> %s", str(row.get('title', ''))[:55], reason)
                    result = skipped_result(reason)
                    evaluated = True  # deterministic skip — mark seen
                else:
                    result, evaluated = evaluate_job_with_ai(row, cv_text, cerebras_key, groq_key)

                    # Layer 3: live remote-eligibility verification via Gemini + Google Search.
                    # Only fires for AI-validated rows where the geo signals are ambiguous.
                    # On "restricted" the verdict's is_valid is flipped to False, so the row
                    # drops out of the email below at the valid_mask filter.
                    if evaluated and result.get("is_valid") and should_geo_check(row):
                        geo_result = check_remote_eligibility(
                            title=str(row.get("title", "")),
                            company=str(row.get("company", "")),
                            job_url=str(row.get("job_url", "")),
                            description=str(row.get("description", "")),
                            api_key=gemini_key,
                        )
                        result = apply_geo_check_result(result, geo_result)
                        geo_checked += 1

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
                # Only mark seen when AI actually returned a verdict. Quota/timeout
                # errors leave the job unmarked so we can retry it next run.
                if evaluated:
                    tracker.mark_seen(str(row.get("job_url", "")))

            logger.info("Pre-screen summary: skipped %d / %d jobs before AI eval.", prescreen_skipped, len(combined_jobs))
            logger.info("Geo-check summary (top section): %d / %d AI-validated jobs checked.", geo_checked, len(combined_jobs))
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
            
            # Filter down to only AI-approved jobs
            combined_jobs = combined_jobs[valid_mask]
            stats['approved'] = len(combined_jobs)
            logger.info("Total jobs remaining after AI validation: %d", stats['approved'])

            # Lower-ranked geo-check loop. These jobs never got an AI verdict,
            # so the geo-check is their primary geographic safety filter.
            # We do the same deterministic should_geo_check trigger, then ask
            # Gemini live whether a Palestine-based candidate can apply.
            #   restricted -> dropped from the lower-ranked list silently
            #   uncertain  -> kept, but flagged so the rendering shows a warning
            #   open       -> kept, flagged as verified
            if not lower_ranked.empty and gemini_key:
                logger.info("Running geo-eligibility check on %d lower-ranked jobs...", len(lower_ranked))
                geo_statuses = []
                geo_evidences = []
                keep_mask = []
                lr_checked = 0
                for _, lr_row in lower_ranked.iterrows():
                    if should_geo_check(lr_row):
                        geo = check_remote_eligibility(
                            title=str(lr_row.get("title", "")),
                            company=str(lr_row.get("company", "")),
                            job_url=str(lr_row.get("job_url", "")),
                            description=str(lr_row.get("description", "")),
                            api_key=gemini_key,
                        )
                        lr_checked += 1
                        eligibility = geo.get("eligibility", "uncertain")
                        if eligibility == "restricted":
                            keep_mask.append(False)
                            geo_statuses.append("restricted")
                            geo_evidences.append(geo.get("evidence", ""))
                            continue
                        keep_mask.append(True)
                        geo_statuses.append(eligibility)
                        geo_evidences.append(geo.get("evidence", ""))
                    else:
                        # No geo trigger fired — job already explicitly worldwide.
                        keep_mask.append(True)
                        geo_statuses.append("open")
                        geo_evidences.append("")
                lower_ranked = lower_ranked.copy()
                lower_ranked['geo_status'] = geo_statuses
                lower_ranked['geo_evidence'] = geo_evidences
                lower_ranked = lower_ranked[keep_mask].reset_index(drop=True)
                logger.info(
                    "Lower-ranked geo-check: checked %d, dropped %d as restricted, %d remain.",
                    lr_checked, sum(1 for s in geo_statuses if s == "restricted"), len(lower_ranked),
                )

            # 5. Output and Notification
            # Split into internships and jobs
            intern_mask = combined_jobs['title'].str.lower().str.contains('intern') | (combined_jobs.get('job_type', pd.Series(dtype=str)).astype(str).str.lower().str.contains('internship'))
            internships_df = combined_jobs[intern_mask]
            jobs_df = combined_jobs[~intern_mask]
            
            output_config = config.get("output", {"use_email": True, "use_github_issue": False})
            
            if output_config.get("use_email"):
                html_content = format_email_html(internships_df, jobs_df, stats, lower_ranked_df=lower_ranked)
                send_email("Your Automated AI Job Alerts", html_content, config.get("email_settings", {}))

            if output_config.get("use_github_issue"):
                md_content = format_github_markdown(internships_df, jobs_df, stats, lower_ranked_df=lower_ranked)
                today = datetime.now().strftime("%Y-%m-%d")
                # Route new issues to the private logs repo when LOGS_REPO_* are set;
                # when missing (e.g. local dev), fall back to the working repo as before.
                create_github_issue(
                    f"Automated AI Job Alerts - {today}",
                    md_content,
                    repo=os.environ.get("LOGS_REPO"),
                    token=os.environ.get("LOGS_REPO_TOKEN"),
                )
        else:
            logger.info("No new jobs survived the filters today.")
    else:
        logger.warning("No job data collected from any sources.")

if __name__ == "__main__":
    main()
