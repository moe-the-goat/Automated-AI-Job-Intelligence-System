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
from pipeline.core_ai import evaluate_job_with_ai, evaluate_job_with_gemini, quick_viability_check, skipped_result
from pipeline.core_embedding import attach_similarity, drop_semantic_duplicates, update_embedding_history
from pipeline.core_notify import format_email_html, format_github_markdown, send_email, create_github_issue, cleanup_old_github_issues
from pipeline.core_feedback import ingest_pending_feedback, load_candidate_preferences
from pipeline.core_feedback_page import render_feedback_page, write_feedback_page

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

# A3: only the top N jobs (by weighted CV-embedding similarity) get the
# Cerebras+Groq verdict (slower, larger model, better verdicts). Ranking
# multiplies raw similarity by region+trust+role weights so EU / Americas /
# Middle East and trusted big-name companies sort above India-only postings at
# equal raw similarity. A wildcard sample is also evaluated so an imperfectly-
# tuned weighting doesn't permanently hide a long-tail match.
AI_EVAL_TOP_N = 55
WILDCARD_COUNT = 5

# Lower-ranked jobs (the rest, after the top N+wildcards) get a cheaper Gemini
# 3.1 Flash Lite verdict so they appear with real AI judgements in the email's
# "Also Found" section instead of just a similarity score. We cap how many we
# evaluate so the run doesn't balloon past the cron budget — Gemini Flash Lite
# is throttled at 15 RPM (~4s/call), so 25 calls = ~1.7 min of throttle plus
# response time. Display still caps at 15 after filtering is_valid=False rows.
LOWER_RANKED_EVAL_LIMIT = 25


def _run_ai_loop(df, cv_text, tracker, *, provider, cerebras_key=None, groq_key=None, gemini_key=None, learned_preferences=""):
    """Run quick_viability_check + AI verdict over a dataframe of jobs.

    Returns the SAME dataframe with these columns attached:
      ai_verdict, match_percentage, tech_fit, experience_fit, logistics_fit,
      compensation, effort, suspicious, scam, is_valid, pre_flagged_low_quality

    `provider` chooses which LLM:
      "cerebras_groq" -> evaluate_job_with_ai (Qwen-3 + Llama-3.3 ping-pong)
      "gemini"        -> evaluate_job_with_gemini (Flash Lite, single call + retries)

    Tracker is marked seen only when the LLM actually returned a verdict, so
    transient errors leave the job unmarked for next-run retry.
    """
    if df is None or df.empty:
        return df

    verdicts, valid_mask, match_pcts = [], [], []
    tech_fits, exp_fits, log_fits = [], [], []
    comps, efforts, suspiciouses, scams = [], [], [], []
    blacklisteds = []
    prescreen_skipped = 0

    for _, row in df.iterrows():
        is_viable, reason = quick_viability_check(row)
        if not is_viable:
            prescreen_skipped += 1
            logger.info("[SKIP] %-55s -> %s", str(row.get('title', ''))[:55], reason)
            result = skipped_result(reason)
            evaluated = True  # deterministic skip — mark seen
        else:
            if provider == "gemini":
                result, evaluated = evaluate_job_with_gemini(row, cv_text, gemini_key, learned_preferences=learned_preferences)
            else:
                result, evaluated = evaluate_job_with_ai(row, cv_text, cerebras_key, groq_key, learned_preferences=learned_preferences)

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
        if evaluated:
            tracker.mark_seen(str(row.get("job_url", "")))

    logger.info("Pre-screen summary (%s): skipped %d / %d jobs before AI eval.",
                provider, prescreen_skipped, len(df))
    out = df.copy()
    out['ai_verdict'] = verdicts
    out['match_percentage'] = match_pcts
    out['tech_fit'] = tech_fits
    out['experience_fit'] = exp_fits
    out['logistics_fit'] = log_fits
    out['compensation'] = comps
    out['effort'] = efforts
    out['suspicious'] = suspiciouses
    out['scam'] = scams
    out['pre_flagged_low_quality'] = blacklisteds
    out['is_valid'] = valid_mask
    return out


def run_pipeline(config, tracker):
    all_jobs_dfs = []

    # Drain yesterday's feedback before anything else: hard signals
    # (block_company, applied) must influence today's run.
    logs_repo = os.environ.get("LOGS_REPO")
    logs_token = os.environ.get("LOGS_REPO_TOKEN")
    ingest_pending_feedback(logs_repo, logs_token, tracker)
    learned_preferences = load_candidate_preferences(logs_repo, logs_token)
    if learned_preferences:
        logger.info("Loaded candidate preferences (%d chars) for verdict context.", len(learned_preferences))

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
            #  - Gemini EMBED key: dedicated to embedding-pre-rank only. We split keys
            #    because the embedding burst (~100 RPM peak on ~120 jobs/day) was
            #    poisoning the shared project quota. Falls back to the main key
            #    if EMBED_API_KEY isn't configured.
            #  - Gemini main key: used by core_ai.evaluate_job_with_gemini for the
            #    lower-ranked "Also Found" verdicts (cheap second-pass AI).
            #  - Cerebras/Groq keys: used by core_ai.evaluate_job_with_ai for the
            #    top-section verdict (larger Qwen/Llama models, ping-pong fallback).
            gemini_key = os.environ.get("GEMINI_API_KEY", "")
            gemini_embed_key = os.environ.get("GEMINI_EMBED_API_KEY", "") or gemini_key
            cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
            groq_key = os.environ.get("GROQ_API_KEY", "")
            try:
                with open("cv_text.txt", "r", encoding="utf-8") as f:
                    cv_text = f.read()
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("cv_text.txt unreadable (%s) — using built-in fallback CV.", e)
                cv_text = "Computer Engineering student specializing in AI systems engineering, building end-to-end pipelines that integrate LLMs, embeddings, and multi-source data into production-ready backend systems. Experienced deploying Python-based solutions with REST APIs, automated workflows, and real-world constraints. Growing focus on Generative AI, RAG architectures, and scalable intelligent systems."

            # A3: pre-rank by CV-embedding similarity. Top-N + wildcards go to
            # the Cerebras+Groq path; the rest get Gemini in the lower-ranked
            # path. attach_similarity also returns the raw job embeddings so we
            # can run semantic dedup against the rolling 14-day history cache.
            combined_jobs, job_embeddings = attach_similarity(combined_jobs, cv_text, gemini_embed_key)

            # Semantic dedup: drop jobs whose embedding is ≥0.97 cosine similar
            # to anything in the embedding history (last 14 days). Catches the
            # "same job reposted with a different URL" case that URL dedup misses.
            before = len(combined_jobs)
            combined_jobs = drop_semantic_duplicates(combined_jobs, job_embeddings)
            dropped_semantic = before - len(combined_jobs)
            if dropped_semantic:
                logger.info("Semantic dedup: dropped %d job(s) ≥0.97 similar to recent history.", dropped_semantic)

            if len(combined_jobs) > AI_EVAL_TOP_N:
                top_n = combined_jobs.head(AI_EVAL_TOP_N)
                rest = combined_jobs.iloc[AI_EVAL_TOP_N:]
                n_wild = min(WILDCARD_COUNT, len(rest))
                wildcards = rest.sample(n_wild, random_state=42) if n_wild > 0 else rest.iloc[:0]
                ai_eval_set = pd.concat([top_n, wildcards]).reset_index(drop=True)
                lower_ranked = rest.drop(wildcards.index).reset_index(drop=True)
                logger.info(
                    "Embedding pre-rank: %d jobs to Cerebras/Groq (top %d + %d wildcards); %d lower-ranked deferred to Gemini.",
                    len(ai_eval_set), AI_EVAL_TOP_N, n_wild, len(lower_ranked),
                )
            else:
                ai_eval_set = combined_jobs
                lower_ranked = pd.DataFrame()
                logger.info("Embedding pre-rank: %d jobs (under threshold, evaluating all with Cerebras/Groq).", len(ai_eval_set))

            # --- Top section: Cerebras+Groq verdicts ---
            logger.info("Running top-section AI validation via Cerebras+Groq (this may take a while)...")
            ai_eval_set = _run_ai_loop(ai_eval_set, cv_text, tracker,
                                       provider="cerebras_groq",
                                       cerebras_key=cerebras_key, groq_key=groq_key,
                                       learned_preferences=learned_preferences)
            valid_top = ai_eval_set[ai_eval_set['is_valid']]
            stats['approved'] = len(valid_top)
            logger.info("Total top-section jobs remaining after AI validation: %d", stats['approved'])

            # --- Lower-ranked section: Gemini 3.1 Flash Lite verdicts ---
            if not lower_ranked.empty and gemini_key:
                eval_slice = lower_ranked.head(LOWER_RANKED_EVAL_LIMIT).reset_index(drop=True)
                deferred_count = len(lower_ranked) - len(eval_slice)
                logger.info(
                    "Running lower-ranked AI validation via Gemini for %d job(s) (%d deferred over cap).",
                    len(eval_slice), deferred_count,
                )
                eval_slice = _run_ai_loop(eval_slice, cv_text, tracker,
                                          provider="gemini",
                                          gemini_key=gemini_key,
                                          learned_preferences=learned_preferences)
                lower_ranked = eval_slice[eval_slice['is_valid']].reset_index(drop=True)
                logger.info("Lower-ranked jobs surviving Gemini validation: %d", len(lower_ranked))
            else:
                if lower_ranked.empty:
                    logger.info("Lower-ranked section: no jobs below the embedding threshold.")
                else:
                    logger.warning(
                        "Lower-ranked section: skipping %d job(s) — GEMINI_API_KEY not set.",
                        len(lower_ranked),
                    )
                lower_ranked = pd.DataFrame()

            # Persist the embeddings we just computed (only for jobs that
            # survived all filters) so the next run's semantic dedup has fresh
            # history to compare against.
            kept_urls = set(valid_top.get('job_url', pd.Series(dtype=str)).tolist()) | \
                        set(lower_ranked.get('job_url', pd.Series(dtype=str)).tolist())
            update_embedding_history({u: v for u, v in job_embeddings.items() if u in kept_urls})

            combined_jobs = valid_top

            # 5. Output and Notification
            # Split into internships and jobs
            intern_mask = combined_jobs['title'].str.lower().str.contains('intern') | (combined_jobs.get('job_type', pd.Series(dtype=str)).astype(str).str.lower().str.contains('internship'))
            internships_df = combined_jobs[intern_mask]
            jobs_df = combined_jobs[~intern_mask]
            
            # Generate the feedback page BEFORE dispatch so the email link
            # always resolves to the freshly-rendered jobs of this run. The
            # page is rendered unconditionally — a stale page showing yesterday's
            # jobs is worse than a fresh page whose submit button is disabled,
            # and the template already reports "missing worker URL" on submit
            # when the env var hasn't been wired up yet.
            feedback_worker_url = os.environ.get("FEEDBACK_WORKER_URL", "")
            feedback_path = os.environ.get("FEEDBACK_PAGE_PATH", "docs/feedback_global.html")
            if not feedback_worker_url:
                logger.warning("FEEDBACK_WORKER_URL missing — rendering page without a working submit button.")
            try:
                page_html = render_feedback_page(
                    dfs=[internships_df, jobs_df, lower_ranked],
                    worker_url=feedback_worker_url,
                    page_title="Job Feedback (Global)",
                )
                write_feedback_page(feedback_path, page_html)
            except Exception as e:
                logger.warning("Feedback page generation failed: %s", e)

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
