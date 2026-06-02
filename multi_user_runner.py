"""
MULTI-USER RUNNER — B7
----------------------
The multi-tenant analog of `scraper.py`. Wakes up on a fixed cron tick,
loads every user whose `preferences.next_run_at <= NOW()` and `is_active`,
runs the existing pipeline against each one's CV + search_queries, persists
results into Supabase, and dispatches the daily email via Resend.

Stays a separate top-level entry point (next to scraper.py and
local_companies.py) rather than refactoring scraper.run_pipeline() so:

  * the single-user pipeline keeps shipping mail daily during the B10
    dual-run window without us holding two contracts in our head, and
  * the multi-user worker can diverge as it needs to (skip cv_text.txt,
    skip GitHub-Issue routing, skip the static feedback page) without
    breaking the single-user path.

CLI:
    python multi_user_runner.py                      # process all due users
    python multi_user_runner.py --dry-run            # persist runs+results, no email
    python multi_user_runner.py --user-id <uuid>     # process one specific user
    python multi_user_runner.py --skip-due-check     # ignore next_run_at gating
                                                       (use with --user-id for forced replays)

Env (in addition to the scraper's existing secrets):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    RESEND_API_KEY
    GEMINI_EMBED2_API_KEY   (or GEMINI_EMBED_API_KEY / GEMINI_API_KEY fallback)
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from pipeline.logging_setup import configure_logging, get_logger
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
from pipeline.core_filter import filter_api_jobs, apply_pipeline_filters
from pipeline.core_ai import (
    evaluate_job_with_ai,
    evaluate_job_with_gemini,
    quick_viability_check,
    skipped_result,
)
from pipeline.core_embedding import (
    attach_similarity,
    drop_semantic_duplicates,
    retrieve_relevant_feedback,
)
from pipeline.core_notify import format_email_html
from pipeline.core_email_resend import send_email as send_via_resend
from pipeline.core_supabase import (
    SupabaseConfigError,
    SupabaseJobTracker,
    get_service_client,
    load_job_embedding_history,
    save_job_embedding_history,
)
from pipeline.core_feedback_supabase import (
    RAG_FEEDBACK_THRESHOLD,
    RAG_TOP_K,
    count_feedback_entries,
    ensure_feedback_embeddings,
    load_candidate_preferences,
    load_feedback_embeddings,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tuning knobs (mirror scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

AI_EVAL_TOP_N = 30                  # smaller default per-user budget than the global scraper (55)
WILDCARD_COUNT = 5
LOWER_RANKED_EVAL_LIMIT = 15        # smaller than scraper.py (25) — multiplied by N users
DESCRIPTION_EXCERPT_CHARS = 1000    # persisted to job_results so Tab B survives URL rot

# Closed-beta gate: only users with profiles.is_whitelisted = true are
# processed. The worker is the thing that spends API quota and sends mail, so
# the whitelist is enforced HERE (not just in the UI). Set False to open up.
WHITELIST_ONLY = True

# Run lock: a user's next_run_at is pushed forward to now + cadence at the START
# of their run (a claim), so an overlapping cron tick can't re-select a user
# whose run is still in flight. On failure we pull it back to a shorter retry.
FAILURE_RETRY_HOURS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight per-tick caches
# ─────────────────────────────────────────────────────────────────────────────
#
# Public-API fetchers (Remotive / Arbeitnow / etc.) take no arguments and
# return the entire feed. With N users we'd be re-fetching the same payload
# N times per cron tick. Cache by name within a single run() invocation so
# 5 users = 1 fetch.
class _ApiCache:
    def __init__(self):
        self._cache: dict = {}

    def get(self, name: str, fetch_fn):
        if name not in self._cache:
            try:
                self._cache[name] = fetch_fn()
                logger.info("API cache: %s fetched (%d rows).", name, len(self._cache[name]))
            except Exception as e:
                logger.warning("API cache: %s fetch failed: %s. Treating as empty.", name, e)
                self._cache[name] = pd.DataFrame()
        return self._cache[name]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _run_ai_loop(df, cv_text, tracker, *,
                 provider, cerebras_key=None, groq_key=None,
                 gemini_key=None, preferences_for=None):
    """Same shape as scraper._run_ai_loop, copied (not imported) so we don't
    introduce an entry-point-to-entry-point dependency. See scraper.py for
    the full design notes — the contract is identical.
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
            evaluated = True
        else:
            prefs = preferences_for(row) if preferences_for else ""
            if provider == "gemini":
                result, evaluated = evaluate_job_with_gemini(
                    row, cv_text, gemini_key, learned_preferences=prefs
                )
            else:
                result, evaluated = evaluate_job_with_ai(
                    row, cv_text, cerebras_key, groq_key, learned_preferences=prefs
                )

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

    logger.info("Pre-screen summary (%s): skipped %d / %d.",
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


def _scrape_for_user(user_row: dict, api_cache: _ApiCache) -> pd.DataFrame:
    """Run all search sources for one user, return a combined raw DataFrame.

    JobSpy queries are per-user (each user has different search_terms +
    locations). The public-API feeds are global, so they come from the
    cache shared across users on this cron tick.
    """
    from jobspy import scrape_jobs  # lazy: heavy import only when actually scraping

    all_dfs = []

    # 1) Per-user JobSpy searches
    for search in user_row.get("search_queries", []):
        term = search.get("search_term", "")
        location = search.get("location", "Worldwide")
        try:
            jobs = scrape_jobs(
                site_name=search.get("sites") or ["linkedin", "indeed"],
                search_term=term,
                location=location,
                distance=search.get("distance", 50),
                job_type=search.get("job_type"),
                is_remote=bool(search.get("is_remote", True)),
                results_wanted=int(search.get("results_wanted", 30)),
                hours_old=int(search.get("hours_old", 24)),
                country_indeed=search.get("country_indeed", "USA"),
            )
            logger.info("JobSpy '%s' in '%s': %d jobs.", term, location, len(jobs))
            all_dfs.append(jobs)
        except Exception as e:
            logger.warning("JobSpy '%s' in '%s' failed: %s", term, location, e)

    # 2) Shared public APIs (cached across users this tick)
    api_hours_old = int(user_row.get("api_hours_old", 72))
    for name, fetch_fn in [
        ("Remotive", fetch_remotive_jobs),
        ("Arbeitnow", fetch_arbeitnow_jobs),
        ("Jobicy", fetch_jobicy_jobs),
        ("RemoteOK", fetch_remoteok_jobs),
        ("Himalayas", fetch_himalayas_jobs),
        ("TheMuse", fetch_themuse_jobs),
        ("WWR", fetch_wwr_jobs),
    ]:
        raw = api_cache.get(name, fetch_fn)
        if raw.empty:
            continue
        filtered = filter_api_jobs(raw, hours_old=api_hours_old)
        if not filtered.empty:
            all_dfs.append(filtered)

    # 3) YC Work at a Startup — only if Gemini key is set
    yc_key = os.environ.get("GEMINI_API_KEY", "")
    if yc_key:
        yc_raw = api_cache.get("YC", lambda: fetch_yc_workatastartup_jobs(gemini_api_key=yc_key))
        if not yc_raw.empty:
            all_dfs.append(yc_raw)

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def _build_preferences_provider(user_id: str, feedback_embed_key: str):
    """Decide RAG vs digest for one user, return (preferences_for_callable, mode_label)."""
    entry_count = ensure_feedback_embeddings(user_id, feedback_embed_key)

    if entry_count >= RAG_FEEDBACK_THRESHOLD:
        embeddings = load_feedback_embeddings(user_id)
        logger.info("User %s: RAG mode (%d feedback entries).", user_id, entry_count)

        def preferences_for(row):
            return retrieve_relevant_feedback(row, embeddings, feedback_embed_key, top_k=RAG_TOP_K)

        return preferences_for, "rag"

    digest = load_candidate_preferences(user_id)
    logger.info(
        "User %s: digest mode (%d entries; RAG at %d). Profile: %s.",
        user_id, entry_count, RAG_FEEDBACK_THRESHOLD,
        f"{len(digest)} chars" if digest else "empty",
    )

    def preferences_for(_row):
        return digest

    return preferences_for, "digest"


# ─────────────────────────────────────────────────────────────────────────────
# Supabase persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_due_users(client, *, only_user_id: Optional[str] = None, skip_due_check: bool = False):
    """Load preferences + profile + search_queries for users due to run.

    Returns a list of dicts shaped:
      {
        "user_id": uuid,
        "notification_email": str,
        "cv_text": str,
        "frequency_hours": int,
        "api_hours_old": int,
        "ai_eval_top_n": int,
        "search_queries": [ {search_term, location, sites, ...}, ... ],
      }
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Query 1: due preferences + the 1:1 profile. We can embed profiles because
    # preferences.user_id is an FK to profiles. We CANNOT embed search_queries
    # here — it has no FK to preferences (both only reference profiles), so
    # PostgREST rejects the join (PGRST200). search_queries is loaded separately.
    query = (
        client.table("preferences")
        .select(
            "user_id, frequency_hours, is_active, next_run_at, "
            "notification_email, ai_eval_top_n, api_hours_old, "
            "profiles!inner(cv_text, is_whitelisted)"
        )
        .eq("is_active", True)
    )

    if only_user_id:
        query = query.eq("user_id", only_user_id)
    elif not skip_due_check:
        query = query.lte("next_run_at", now_iso)

    try:
        resp = query.execute()
    except Exception as e:
        logger.critical("Failed to load due users: %s", e)
        return []

    pref_rows = resp.data or []
    if not pref_rows:
        return []

    # Query 2: active search_queries for exactly those users, in one round trip.
    user_ids = [r["user_id"] for r in pref_rows]
    searches_by_user: dict[str, list] = {}
    try:
        searches_resp = (
            client.table("search_queries")
            .select(
                "user_id, search_term, location, sites, job_type, is_remote, "
                "results_wanted, hours_old, country_indeed, is_active"
            )
            .in_("user_id", user_ids)
            .eq("is_active", True)
            .execute()
        )
        for s in searches_resp.data or []:
            searches_by_user.setdefault(s["user_id"], []).append(s)
    except Exception as e:
        logger.critical("Failed to load search_queries: %s", e)
        return []

    users = []
    for row in pref_rows:
        uid = row["user_id"]
        profile = row.get("profiles") or {}
        if isinstance(profile, list):
            profile = profile[0] if profile else {}

        # Closed-beta gate — enforced at the worker, not just the UI.
        if WHITELIST_ONLY and not profile.get("is_whitelisted"):
            logger.info("Skipping user %s — not whitelisted (closed beta).", uid)
            continue

        cv_text = (profile.get("cv_text") or "").strip()
        if not cv_text:
            logger.warning("Skipping user %s — no cv_text on profile.", uid)
            continue

        searches = searches_by_user.get(uid, [])
        if not searches:
            logger.warning("Skipping user %s — no active search_queries.", uid)
            continue

        users.append({
            "user_id": uid,
            "notification_email": row["notification_email"],
            "cv_text": cv_text,
            "frequency_hours": row["frequency_hours"],
            "api_hours_old": row["api_hours_old"],
            "ai_eval_top_n": row.get("ai_eval_top_n") or AI_EVAL_TOP_N,
            "search_queries": searches,
        })
    return users


def _insert_run(client, user_id: str) -> Optional[int]:
    """Create the runs row with status='running' and return its id."""
    try:
        resp = (
            client.table("runs")
            .insert({"user_id": user_id, "status": "running"})
            .execute()
        )
        rows = resp.data or []
        return rows[0]["id"] if rows else None
    except Exception as e:
        logger.error("INSERT runs failed for user %s: %s", user_id, e)
        return None


def _finalize_run(client, run_id: int, *,
                  status: str,
                  scraped: int = 0, filtered: int = 0,
                  ai_evaluated: int = 0, approved: int = 0,
                  lower_ranked: int = 0, error: Optional[str] = None):
    try:
        client.table("runs").update({
            "status": status,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "scraped": scraped,
            "filtered": filtered,
            "ai_evaluated": ai_evaluated,
            "approved": approved,
            "lower_ranked": lower_ranked,
            "error": (error or "")[:2000] if error else None,
        }).eq("id", run_id).execute()
    except Exception as e:
        logger.error("UPDATE runs failed for run %s: %s", run_id, e)


def _persist_job_results(client, run_id: int, user_id: str,
                         top_df: pd.DataFrame, lower_df: pd.DataFrame):
    """Bulk-insert job_results for one run. ai_evaluated=True for top + wildcards;
    ai_evaluated=False for the lower-ranked section."""
    rows = []
    rows.extend(_jobs_to_rows(top_df, run_id, user_id, ai_evaluated=True))
    rows.extend(_jobs_to_rows(lower_df, run_id, user_id, ai_evaluated=False))
    if not rows:
        return 0
    try:
        # Chunk to keep payload size sane on big runs.
        chunk = 100
        inserted = 0
        for i in range(0, len(rows), chunk):
            client.table("job_results").insert(rows[i:i + chunk]).execute()
            inserted += len(rows[i:i + chunk])
        return inserted
    except Exception as e:
        logger.error("Bulk INSERT job_results failed for run %s: %s", run_id, e)
        return 0


def _jobs_to_rows(df: pd.DataFrame, run_id: int, user_id: str, *, ai_evaluated: bool):
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        description = str(row.get("description", "") or "")
        out.append({
            "run_id": run_id,
            "user_id": user_id,
            "title": _safe_str(row.get("title")),
            "company": _safe_str(row.get("company")),
            "location": _safe_str(row.get("location")),
            "job_url": _safe_str(row.get("job_url")),
            "description_excerpt": description[:DESCRIPTION_EXCERPT_CHARS] or None,
            "ai_evaluated": ai_evaluated,
            "ai_verdict": _safe_str(row.get("ai_verdict")) if ai_evaluated else None,
            "is_valid": _coerce_bool(row.get("is_valid")) if ai_evaluated else None,
            "match_percentage": _coerce_int(row.get("match_percentage")),
            "tech_fit": _coerce_int(row.get("tech_fit")),
            "experience_fit": _coerce_int(row.get("experience_fit")),
            "logistics_fit": _coerce_int(row.get("logistics_fit")),
            "compensation": _safe_str(row.get("compensation")),
            "effort": _safe_effort(row.get("effort")),
            "suspicious": bool(row.get("suspicious", False)),
            "pre_flagged_low_quality": bool(row.get("pre_flagged_low_quality", False)),
            "pre_flagged_trusted": bool(row.get("pre_flagged_trusted", False)),
            "similarity": _coerce_similarity(row.get("similarity")),
        })
    return out


def _safe_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _coerce_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return None
    if not (0 <= n <= 100):
        # The schema CHECK is 0..100; clamp instead of dropping.
        return max(0, min(100, n))
    return n


def _coerce_bool(v) -> Optional[bool]:
    if v is None:
        return None
    return bool(v)


def _coerce_similarity(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, min(1.0, f)), 4)


_VALID_EFFORTS = {"low", "medium", "high", "unknown"}


def _safe_effort(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s in _VALID_EFFORTS else None


def _compute_next_run(frequency_hours, *, now=None) -> datetime:
    """When the user should next be eligible: now + their cadence (min 1h)."""
    now = now or datetime.now(timezone.utc)
    return now + timedelta(hours=max(1, int(frequency_hours)))


def _compute_retry(*, now=None) -> datetime:
    """When to retry after a failed run — sooner than a full cadence so a
    transient error (provider 5xx, network) recovers without waiting a day."""
    now = now or datetime.now(timezone.utc)
    return now + timedelta(hours=FAILURE_RETRY_HOURS)


def _set_next_run(client, user_id: str, when: datetime):
    try:
        client.table("preferences").update(
            {"next_run_at": when.isoformat()}
        ).eq("user_id", user_id).execute()
    except Exception as e:
        logger.error("UPDATE preferences.next_run_at failed for %s: %s", user_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# Per-user pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_for_user(user: dict, client, api_cache: _ApiCache, *, dry_run: bool):
    """End-to-end pipeline for one user. Never raises — failures are logged
    and recorded on the runs row so other users in the batch still execute.
    """
    user_id = user["user_id"]
    run_id = _insert_run(client, user_id)
    if run_id is None:
        logger.error("Could not create run row for %s — skipping.", user_id)
        return

    # Claim the user up front: push next_run_at to now + cadence BEFORE the slow
    # pipeline runs, so a cron tick that overlaps this run won't re-select this
    # user mid-flight (the workflow `concurrency` guard is the first line of
    # defense; this is the data-layer backstop, e.g. for a manual dispatch that
    # overlaps the schedule). On failure we pull it back to a short retry below.
    _set_next_run(client, user_id, _compute_next_run(user["frequency_hours"]))

    tracker = SupabaseJobTracker(user_id, client=client)
    stats = {"scraped": 0, "filtered": 0, "ai_evaluated": 0, "approved": 0, "lower_ranked": 0}

    try:
        # 1. Build the RAG / digest preferences provider
        feedback_embed_key = (
            os.environ.get("GEMINI_EMBED2_API_KEY")
            or os.environ.get("GEMINI_EMBED_API_KEY")
            or os.environ.get("GEMINI_API_KEY", "")
        )
        preferences_for, _mode = _build_preferences_provider(user_id, feedback_embed_key)

        # 2. Scrape
        combined = _scrape_for_user(user, api_cache)
        stats["scraped"] = int(len(combined))
        if combined.empty:
            logger.info("User %s: no jobs scraped this tick.", user_id)
            _finalize_run(client, run_id, status="success", **stats)
            tracker.save()
            return

        # 3. Deterministic filters (tracker drops seen URLs first)
        combined = apply_pipeline_filters(combined, tracker=tracker)
        stats["filtered"] = int(len(combined))
        if combined.empty:
            logger.info("User %s: 0 jobs survived the pre-filters.", user_id)
            _finalize_run(client, run_id, status="success", **stats)
            tracker.save()
            return

        # 4. CV embedding pre-rank
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        gemini_embed_key = os.environ.get("GEMINI_EMBED_API_KEY", "") or gemini_key
        cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")

        combined, job_embeddings = attach_similarity(
            combined, user["cv_text"], gemini_embed_key
        )

        # 4b. Semantic dedup — drop "same job reposted at a new URL" against this
        # user's rolling 14-day embedding history (the multi-user analog of the
        # legacy local cache; SupabaseJobTracker already handled exact-URL repeats).
        history = load_job_embedding_history(user_id, client=client)
        if history:
            before = len(combined)
            combined = drop_semantic_duplicates(combined, job_embeddings, history=history)
            dropped = before - len(combined)
            if dropped:
                logger.info("User %s: semantic dedup dropped %d reposted job(s).", user_id, dropped)

        # 5. Split into AI-eval set + lower-ranked
        top_n = int(user.get("ai_eval_top_n") or AI_EVAL_TOP_N)
        if len(combined) > top_n:
            top_slice = combined.head(top_n)
            rest = combined.iloc[top_n:]
            n_wild = min(WILDCARD_COUNT, len(rest))
            wildcards = rest.sample(n_wild, random_state=42) if n_wild else rest.iloc[:0]
            ai_set = pd.concat([top_slice, wildcards]).reset_index(drop=True)
            lower_ranked = rest.drop(wildcards.index).reset_index(drop=True)
        else:
            ai_set = combined.reset_index(drop=True)
            lower_ranked = pd.DataFrame()

        stats["ai_evaluated"] = int(len(ai_set))

        # 6. Top section — Cerebras+Groq
        ai_set = _run_ai_loop(
            ai_set, user["cv_text"], tracker,
            provider="cerebras_groq",
            cerebras_key=cerebras_key, groq_key=groq_key,
            preferences_for=preferences_for,
        )
        valid_top = ai_set[ai_set["is_valid"]].reset_index(drop=True)
        stats["approved"] = int(len(valid_top))

        # 7. Lower-ranked — Gemini Flash Lite
        if not lower_ranked.empty and gemini_key:
            eval_slice = lower_ranked.head(LOWER_RANKED_EVAL_LIMIT).reset_index(drop=True)
            eval_slice = _run_ai_loop(
                eval_slice, user["cv_text"], tracker,
                provider="gemini",
                gemini_key=gemini_key,
                preferences_for=preferences_for,
            )
            lower_ranked = eval_slice[eval_slice["is_valid"]].reset_index(drop=True)
        else:
            lower_ranked = pd.DataFrame()
        stats["lower_ranked"] = int(len(lower_ranked))

        # 8. Persist to Supabase
        persisted = _persist_job_results(client, run_id, user_id, valid_top, lower_ranked)
        logger.info("User %s: persisted %d job_results row(s).", user_id, persisted)

        # 9. Send email (skipped on --dry-run)
        if not dry_run and not valid_top.empty:
            intern_mask = (
                valid_top["title"].str.lower().str.contains("intern", na=False)
                | valid_top.get("job_type", pd.Series(dtype=str)).astype(str).str.lower().str.contains("internship", na=False)
            )
            internships_df = valid_top[intern_mask]
            jobs_df = valid_top[~intern_mask]
            html = format_email_html(
                internships_df, jobs_df,
                {"scraped": stats["scraped"], "filtered": stats["filtered"], "approved": stats["approved"]},
                lower_ranked_df=lower_ranked,
            )
            ok, _info = send_via_resend(
                user["notification_email"],
                "Your Daily Job Alerts",
                html,
            )
            if not ok:
                logger.warning("User %s: Resend send failed — marking run success but email skipped.", user_id)
        elif dry_run:
            logger.info("User %s: --dry-run, email not sent.", user_id)

        # Remember the embeddings of jobs that survived to delivery so the next
        # run's semantic dedup has fresh history. Mirror scraper.py: only keep
        # URLs that actually made it into the email (top + lower-ranked).
        kept_urls = set(valid_top.get("job_url", pd.Series(dtype=str)).tolist())
        if not lower_ranked.empty:
            kept_urls |= set(lower_ranked.get("job_url", pd.Series(dtype=str)).tolist())
        survivors = {u: v for u, v in job_embeddings.items() if u in kept_urls and v is not None}
        if survivors:
            save_job_embedding_history(user_id, survivors, client=client)

        _finalize_run(client, run_id, status="success", **stats)
        tracker.save()

    except Exception as e:
        logger.error("User %s: pipeline crashed: %s\n%s", user_id, e, traceback.format_exc())
        _finalize_run(
            client, run_id, status="failed",
            error=f"{type(e).__name__}: {e}",
            **stats,
        )
        # Pull the claim back to a short retry so a transient failure recovers
        # well before the full cadence would have allowed.
        _set_next_run(client, user_id, _compute_retry())
        tracker.save()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-user job-alerts worker (B7).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Persist runs+results to Supabase but do NOT send email.")
    parser.add_argument("--user-id", default=None,
                        help="Process only this user (UUID). Useful for onboarding / replays.")
    parser.add_argument("--skip-due-check", action="store_true",
                        help="Ignore preferences.next_run_at gating (use with --user-id).")
    args = parser.parse_args(argv)

    configure_logging()

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    users = _load_due_users(
        client,
        only_user_id=args.user_id,
        skip_due_check=args.skip_due_check,
    )
    logger.info("Multi-user runner: %d user(s) to process.", len(users))
    if not users:
        return 0

    api_cache = _ApiCache()
    tick_start = time.time()
    for user in users:
        user_start = time.time()
        logger.info("--- Starting user %s ---", user["user_id"])
        _run_for_user(user, client, api_cache, dry_run=args.dry_run)
        logger.info(
            "--- Finished user %s in %.1fs ---",
            user["user_id"], time.time() - user_start,
        )

    logger.info(
        "Multi-user runner: %d user(s) done in %.1fs.",
        len(users), time.time() - tick_start,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
