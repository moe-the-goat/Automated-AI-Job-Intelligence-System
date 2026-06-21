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
    python multi_user_runner.py --user-id <uuid> --skip-due-check --manual
                                                     # user "Run now": stamps the run
                                                       manual + cancels today's scheduled
                                                       tick (still bounded by the 2/day budget)

Env (in addition to the scraper's existing secrets):
    SUPABASE_URL
    SUPABASE_SERVICE_ROLE_KEY
    SENDER_EMAIL            Gmail address that sends the alerts (SMTP transport)
    EMAIL_APP_PASSWORD      Google app password for SENDER_EMAIL
    GEMINI_EMBED2_API_KEY   (or GEMINI_EMBED_API_KEY / GEMINI_API_KEY fallback)
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

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
from pipeline.core_local_sources import collect_local_raw_jobs
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
# Transport: Gmail SMTP (sends to any recipient, no domain needed). Replaced
# Resend, which required a verified custom domain to reach arbitrary addresses.
from pipeline.core_email_smtp import send_email as send_email_transport
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
    create_feedback_token,
    ensure_feedback_embeddings,
    load_candidate_preferences,
    load_feedback_embeddings,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tuning knobs (mirror scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

AI_EVAL_TOP_N = 55                  # matched to the legacy global scraper so the
                                    # finalized-job count is comparable (was 30 —
                                    # too few cleared the bar, ~8-9 vs the old ~20+)
WILDCARD_COUNT = 5
LOWER_RANKED_EVAL_LIMIT = 25        # matched to scraper.py (was 15)
DESCRIPTION_EXCERPT_CHARS = 1000    # persisted to job_results so Tab B survives URL rot

# Closed-beta gate: only users with profiles.is_whitelisted = true are
# processed. The worker is the thing that spends API quota and sends mail, so
# the whitelist is enforced HERE (not just in the UI). Set False to open up.
WHITELIST_ONLY = True

# Run lock: a user's next_run_at is pushed forward to now + cadence at the START
# of their run (a claim), so an overlapping cron tick can't re-select a user
# whose run is still in flight. On failure we pull it back to a shorter retry.
FAILURE_RETRY_HOURS = 2

# Per-user daily run budget. Each user gets this many runs per local day
# (Asia/Jerusalem), counting BOTH the scheduled cron tick and any manual
# "Run now" dispatches — so a user who lets the schedule fire has one manual
# run left, and a user who triggers two manual runs gets no scheduled one.
# Enforced HERE (the worker spends the quota: API tokens + mail) so it can't
# be bypassed by hitting the dispatch endpoint directly; the dashboard only
# mirrors the count for display.
MAX_RUNS_PER_DAY = 2

# The locale the daily budget resets in — matches the project's 9 AM
# Jerusalem schedule, so "today" lines up with the user's day, not UTC.
RUN_BUDGET_TZ = "Asia/Jerusalem"

# Include the Palestinian local-market sources (Telegram, jobs.ps, per-company
# ATS/DDG/JobSpy) in every user's run. The local market is the SAME for everyone,
# so it's collected ONCE per cron tick (shared cache) and merged into each user's
# pipeline, where their own CV-ranking + RAG personalizes which local jobs surface.
# Local rows use the lighter filter (local=True); global rows use the full filter.
INCLUDE_LOCAL_SOURCES = True


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


class _LocalJobsCache:
    """Collects the Palestinian local-market jobs ONCE per cron tick.

    The local market is identical for every user, so we scrape it a single time
    and hand the same raw list to each user's pipeline (where per-user CV ranking
    + RAG decides what surfaces). Lazy: nothing runs until the first user asks.
    """
    def __init__(self):
        self._jobs = None  # None = not yet collected

    def get(self) -> list:
        if self._jobs is None:
            try:
                gemini_key = os.environ.get("GEMINI_API_KEY", "")
                jobs, st = collect_local_raw_jobs(gemini_key=gemini_key)
                self._jobs = jobs
                logger.info(
                    "Local cache: %d raw local job(s) collected this tick (ats=%d ddg=%d jobspy=%d telegram=%d jobs_ps=%d).",
                    len(jobs), st["ats_jobs"], st["ddg_jobs"], st["jobspy_jobs"],
                    st["telegram_jobs"], st["jobsps_jobs"],
                )
            except Exception as e:
                logger.warning("Local cache: collection failed: %s. Treating as empty.", e)
                self._jobs = []
        return self._jobs


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _flush_llm_usage(client, user_id: str) -> None:
    """Persist this user-run's LLM/embedding usage to llm_usage_daily and reset
    the tracker for the next user. Upserts per (user_id, provider, model, day),
    ADDING to today's counters via the increment RPC so multiple runs in a day
    accumulate. Best-effort: any failure is logged, never raised.
    """
    try:
        from pipeline.core_llm_usage import get_tracker
        rows = get_tracker().snapshot_and_reset()
        if not rows:
            # Visible on purpose: "no usage recorded" is itself diagnostic — it
            # means record() never fired (vs. an RPC failure below).
            logger.info("llm_usage: nothing to flush for %s (0 tracked calls).", user_id)
            return
        total = sum(r["requests"] for r in rows)
        logger.info(
            "llm_usage: flushing %d model-rows (%d calls) for %s.",
            len(rows), total, user_id,
        )
        day = _budget_day_str()
        ok_rows = 0
        for r in rows:
            try:
                client.rpc("bump_llm_usage", {
                    "p_user_id": user_id,
                    "p_provider": r["provider"],
                    "p_model": r["model"],
                    "p_day": day,
                    "p_requests": r["requests"],
                    "p_requests_failed": r["requests_failed"],
                    "p_tokens": r["tokens"],
                    "p_peak_rpm": r["peak_rpm"],
                }).execute()
                ok_rows += 1
            except Exception as e:
                logger.warning(
                    "llm_usage flush failed for %s %s/%s: %s",
                    user_id, r["provider"], r["model"], str(e)[:200],
                )
        logger.info("llm_usage: %d/%d rows written for %s.", ok_rows, len(rows), user_id)
    except Exception as e:
        logger.warning("llm_usage flush skipped for %s: %s", user_id, str(e)[:120])


def _join_keys(*env_names) -> str:
    """Comma-join the non-empty values of the named env vars, in order.

    Used to pass MULTIPLE API keys (multiple accounts) for one provider to
    core_llm, which round-robins across them. Returns "" when none are set, "k1"
    for one, "k1,k2" for two — all of which core_llm._parse_keys handles.
    """
    keys = [os.environ.get(n, "").strip() for n in env_names]
    return ",".join(k for k in keys if k)


def _discover_keys(base: str, legacy: str = None) -> str:
    """Auto-discover every API account configured for a provider and comma-join
    them, so adding an account is a pure secrets change — no code edit.

    Discovery order: `base`, then the optional `legacy` name (the historical
    MULTI_* secret that already holds account #2 for Cerebras/Groq), then
    `base_2`, `base_3`, … stopping at the FIRST missing number (numbering must be
    contiguous). Duplicate values are dropped so a key wired under two names isn't
    double-counted. Order doesn't matter — core_llm round-robins over the set.

    Examples:
      _discover_keys("CEREBRAS_API_KEY", "MULTI_CEREBRAS_API_KEY")
        → CEREBRAS_API_KEY, MULTI_CEREBRAS_API_KEY, CEREBRAS_API_KEY_2, _3, …
      _discover_keys("GEMINI_EMBED_API_KEY")
        → GEMINI_EMBED_API_KEY, GEMINI_EMBED_API_KEY_2, _3, …
    """
    names = [base]
    if legacy:
        names.append(legacy)
    n = 2
    while os.environ.get(f"{base}_{n}", "").strip():
        names.append(f"{base}_{n}")
        n += 1

    out, seen = [], set()
    for nm in names:
        v = os.environ.get(nm, "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return ",".join(out)


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
        loaded = len(embeddings.get("entries", []))
        if loaded == 0:
            # entry_count says the corpus is past the RAG threshold, but the
            # embedding join returned nothing. That's not "no feedback" — it's a
            # load failure (e.g. the feedback↔feedback_embeddings FK missing from
            # PostgREST's schema cache, PGRST200). Without this guard the run
            # logs a cheerful "RAG mode (N entries)" while every verdict gets
            # ZERO feedback context. Make the degradation impossible to miss.
            logger.error(
                "User %s: RAG mode selected (%d feedback entries) but the embedding "
                "corpus loaded EMPTY — RAG retrieval will return no context this run. "
                "Check the feedback↔feedback_embeddings relationship / PostgREST schema "
                "cache (see migration 0010).",
                user_id, entry_count,
            )
        else:
            logger.info(
                "User %s: RAG mode (%d feedback entries, %d embedded).",
                user_id, entry_count, loaded,
            )

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
            # The scheduled time that made this user due — used as the ANCHOR for
            # the next run so a late fire doesn't drift the schedule (see
            # _compute_next_run_anchored). May be None for a manual/--skip-due run.
            "next_run_at": row.get("next_run_at"),
        })
    return users


def _insert_run(client, user_id: str, *, trigger: str = "scheduled") -> Optional[int]:
    """Create the runs row with status='running' and return its id.

    `trigger` records whether this was the scheduled cron tick or a manual
    user dispatch (runs.run_trigger). Falls back to a trigger-less insert if
    the column doesn't exist yet (migration 0014 not applied), so a partial
    deploy degrades to the old behavior instead of failing every run.
    """
    if trigger not in ("scheduled", "manual"):
        trigger = "scheduled"
    payload = {"user_id": user_id, "status": "running", "run_trigger": trigger}
    try:
        resp = client.table("runs").insert(payload).execute()
        rows = resp.data or []
        return rows[0]["id"] if rows else None
    except Exception as e:
        # Most likely cause pre-migration: column run_trigger does not exist.
        # Retry once without it so runs keep working during the deploy window.
        logger.warning(
            "INSERT runs with run_trigger failed for %s (%s) — retrying without it.",
            user_id, e,
        )
        try:
            resp = (
                client.table("runs")
                .insert({"user_id": user_id, "status": "running"})
                .execute()
            )
            rows = resp.data or []
            return rows[0]["id"] if rows else None
        except Exception as e2:
            logger.error("INSERT runs failed for user %s: %s", user_id, e2)
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
        # Deploy-order safety net for W1: if the live schema predates
        # migration 0011 (no `origin` column), strip the key and retry once
        # rather than losing the whole run's results. Loud on purpose —
        # the migration should have been applied before this code shipped.
        if "origin" in str(e).lower():
            logger.error(
                "INSERT job_results rejected `origin` for run %s — migration "
                "0011 is NOT applied. Retrying without provenance: %s",
                run_id, e,
            )
            stripped = [{k: v for k, v in r.items() if k != "origin"} for r in rows]
            try:
                chunk = 100
                inserted = 0
                for i in range(0, len(stripped), chunk):
                    client.table("job_results").insert(stripped[i:i + chunk]).execute()
                    inserted += len(stripped[i:i + chunk])
                return inserted
            except Exception as e2:
                logger.error("Retry without origin also failed for run %s: %s", run_id, e2)
                return 0
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
            "origin": _safe_origin(row.get("origin")),
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


_VALID_ORIGINS = {"global", "local"}


def _safe_origin(v):
    """Coerce the provenance tag to the schema's CHECK ('global'|'local').
    Anything else — including pandas NaN, which str()s to 'nan' — maps to
    None so a pre-W1 frame or a malformed value never fails the insert."""
    if v is None:
        return None
    s = str(v).strip().lower()
    return s if s in _VALID_ORIGINS else None


def _compute_next_run(frequency_hours, *, now=None) -> datetime:
    """When the user should next be eligible: now + their cadence (min 1h).

    Used for MANUAL runs (no schedule to anchor to) and as the fallback when a
    scheduled anchor is missing. For SCHEDULED runs use _compute_next_run_anchored
    so a late fire doesn't drift the user's chosen time.
    """
    now = now or datetime.now(timezone.utc)
    return now + timedelta(hours=max(1, int(frequency_hours)))


def _parse_iso(value):
    """Parse an ISO timestamp (Supabase returns these as strings) to an aware
    UTC datetime. Returns None on anything unparseable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _compute_next_run_anchored(scheduled_at, frequency_hours, *, now=None) -> datetime:
    """Next eligible time anchored to the SCHEDULED time, not the fire time.

    A daily user scheduled for 10:00 keeps landing at 10:00 even if a tick fires
    late at 11:00 — the next run is scheduled_at + cadence (10:00 tomorrow), NOT
    fired_at + cadence (11:00 tomorrow). This stops the schedule from drifting
    later every time GitHub/cron fires a bit late.

    Advances in whole cadence steps until strictly in the future, so a very late
    or long-missed run catches up to the next future slot on the ORIGINAL cadence
    rather than firing back-to-back. Falls back to now+cadence when there's no
    usable anchor (manual / first run / unparseable value).
    """
    now = now or datetime.now(timezone.utc)
    step = timedelta(hours=max(1, int(frequency_hours)))
    anchor = _parse_iso(scheduled_at)
    if anchor is None:
        return now + step
    nxt = anchor + step
    # Catch up: if we're so late that anchor+step is already past, keep adding
    # whole steps until the next slot is in the future (bounded loop).
    if nxt <= now:
        # Jump most of the way in one division, then nudge the remainder.
        behind = (now - anchor).total_seconds()
        steps = int(behind // step.total_seconds()) + 1
        nxt = anchor + step * steps
        while nxt <= now:
            nxt += step
    return nxt


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


def _budget_day_start_utc(*, now=None) -> datetime:
    """The UTC instant of the most recent local-midnight in RUN_BUDGET_TZ.

    The daily run budget resets at midnight Jerusalem; runs.started_at is
    stored in UTC, so we convert that local-midnight back to UTC for the
    `started_at >= …` count. Mirrors the runs_used_today SQL RPC exactly so
    the worker's enforcement and the dashboard's display never disagree.
    """
    now = now or datetime.now(timezone.utc)
    tz = ZoneInfo(RUN_BUDGET_TZ)
    local_now = now.astimezone(tz)
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc)


def _budget_day_str(*, now=None) -> str:
    """The LOCAL (Jerusalem) calendar date of the current budget day, as
    YYYY-MM-DD.

    This is the day stamp for llm_usage_daily. Note it must be the LOCAL date,
    not `_budget_day_start_utc().date()` — that returns the UTC date of the
    Jerusalem-midnight instant, which is always the PREVIOUS calendar day (local
    midnight falls at 21:00–22:00 UTC the day before). The web dashboard filters
    "today" by the Jerusalem calendar date, so we must write that same date or
    the Today view is empty all day.
    """
    now = now or datetime.now(timezone.utc)
    return now.astimezone(ZoneInfo(RUN_BUDGET_TZ)).date().isoformat()


def _next_budget_day_start_utc(*, now=None) -> datetime:
    """The UTC instant of the NEXT local-midnight — i.e. when the budget next
    resets. Used to park a user's next_run_at past today so an exhausted or
    manually-triggered user isn't re-selected again today."""
    return _budget_day_start_utc(now=now) + timedelta(days=1)


def _runs_used_today(client, user_id: str, *, now=None) -> int:
    """How many runs this user has started since local-midnight Jerusalem.

    Counts runs rows directly — the same source of truth the dashboard reads
    via the runs_used_today RPC. Returns a large sentinel on query failure so
    a transient DB error fails CLOSED (skips the run) rather than handing out
    free runs; the scheduled tick will simply try again next hour.
    """
    day_start = _budget_day_start_utc(now=now)
    try:
        resp = (
            client.table("runs")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .gte("started_at", day_start.isoformat())
            .execute()
        )
        if resp.count is not None:
            return int(resp.count)
        return len(resp.data or [])
    except Exception as e:
        logger.error("Count runs-today failed for %s: %s — failing closed.", user_id, e)
        return MAX_RUNS_PER_DAY  # treat as exhausted so we don't over-spend


# ─────────────────────────────────────────────────────────────────────────────
# Per-user pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _budget_allows_run(used: int, *, admin_override: bool = False) -> bool:
    """Whether a run may proceed given how many runs the user has already used
    today. An admin override (forced run from /admin) bypasses the 2/day cap;
    otherwise the cap is enforced. Pure helper so the policy is unit-testable."""
    if admin_override:
        return True
    return used < MAX_RUNS_PER_DAY


def _run_for_user(user: dict, client, api_cache: _ApiCache, *, dry_run: bool,
                  local_cache=None, trigger: str = "scheduled",
                  admin_override: bool = False, job_embed_cache=None):
    """End-to-end pipeline for one user. Never raises — failures are logged
    and recorded on the runs row so other users in the batch still execute.

    `local_cache` (a _LocalJobsCache) supplies the shared Palestinian local-market
    jobs collected once per tick. When provided, local jobs are filtered with the
    lighter local=True rules and merged with the user's globally-filtered jobs
    before CV ranking, so each user sees local jobs personalized to their CV.

    `trigger` is 'scheduled' (cron) or 'manual' (a user "Run now" dispatch).
    It's stamped on the runs row and decides the post-run next_run_at handling.
    """
    user_id = user["user_id"]

    # Daily-budget gate. Counts BOTH scheduled and manual runs since local
    # midnight Jerusalem; at/over the cap we skip WITHOUT creating a run row
    # (so the skip itself doesn't burn a slot). This is the authoritative
    # enforcement point — the dashboard's count is only a mirror.
    used = _runs_used_today(client, user_id)
    if not _budget_allows_run(used, admin_override=admin_override):
        logger.info(
            "User %s: daily run budget exhausted (%d/%d) — skipping %s run; "
            "budget resets at local midnight.",
            user_id, used, MAX_RUNS_PER_DAY, trigger,
        )
        # Park a scheduled tick past today so the cron doesn't keep re-selecting
        # this user every hour for the rest of the day. A manual dispatch that
        # hits the cap simply no-ops (the user is told 0 runs left in the UI).
        if trigger == "scheduled":
            _set_next_run(client, user_id, _next_budget_day_start_utc())
        return
    if admin_override and used >= MAX_RUNS_PER_DAY:
        logger.info(
            "User %s: admin override — running despite budget (%d/%d used today).",
            user_id, used, MAX_RUNS_PER_DAY,
        )

    run_id = _insert_run(client, user_id, trigger=trigger)
    if run_id is None:
        logger.error("Could not create run row for %s — skipping.", user_id)
        return

    # Claim the user up front: push next_run_at to now + cadence BEFORE the slow
    # pipeline runs, so a cron tick that overlaps this run won't re-select this
    # user mid-flight (the workflow `concurrency` guard is the first line of
    # defense; this is the data-layer backstop, e.g. for a manual dispatch that
    # overlaps the schedule). On failure we pull it back to a short retry below.
    #
    # On a MANUAL run we additionally cancel today's still-pending scheduled
    # tick: a manual run consumes one of the 2 daily slots, so if the user's
    # next scheduled run is still later TODAY we push it past local midnight.
    # This is what makes "manual now → today's scheduled one is cancelled"
    # hold exactly, instead of the user getting an extra run. (If their cadence
    # already lands tomorrow, next_run_at is left at the normal claim value.)
    if trigger == "manual":
        scheduled_next = _compute_next_run(user["frequency_hours"])
        next_budget_day = _next_budget_day_start_utc()
        _set_next_run(client, user_id, max(scheduled_next, next_budget_day))
    else:
        # Anchor the next run to the user's SCHEDULED time, not when this tick
        # actually fired — so a late fire (e.g. scheduled 10:00, fired 11:00)
        # keeps the next run at 10:00, instead of drifting it to 11:00.
        _set_next_run(
            client, user_id,
            _compute_next_run_anchored(user.get("next_run_at"), user["frequency_hours"]),
        )

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

        # 2. Scrape — global (per-user searches + public APIs) and shared local.
        # Tag provenance HERE, before the merge, so the origin survives the
        # URL-dedup concat and lands in job_results (task W1 — the web app
        # splits Local vs Global/Remote sections on this). The local rows'
        # fine-grained `source` (ddg_linkedin / telegram / jobs_ps …) is
        # collapsed to one 'local' value; source itself stays worker-side.
        global_jobs = _scrape_for_user(user, api_cache)
        if not global_jobs.empty:
            global_jobs["origin"] = "global"
        local_jobs = pd.DataFrame()
        if local_cache is not None:
            raw_local = local_cache.get()
            if raw_local:
                local_jobs = pd.DataFrame(raw_local)
                local_jobs["origin"] = "local"
        stats["scraped"] = int(len(global_jobs) + len(local_jobs))
        if global_jobs.empty and local_jobs.empty:
            logger.info("User %s: no jobs scraped this tick.", user_id)
            _finalize_run(client, run_id, status="success", **stats)
            tracker.save()
            return

        # 3. Deterministic filters (tracker drops seen URLs first). Global jobs
        # get the aggressive firehose filter; local jobs get the lighter local=True
        # filter (pre-vetted companies, Arabic-safe). Then merge for ranking.
        filtered_parts = []
        if not global_jobs.empty:
            g = apply_pipeline_filters(global_jobs, tracker=tracker)
            if not g.empty:
                filtered_parts.append(g)
        if not local_jobs.empty:
            l = apply_pipeline_filters(local_jobs, tracker=tracker, local=True)
            if not l.empty:
                filtered_parts.append(l)

        if not filtered_parts:
            logger.info("User %s: 0 jobs survived the pre-filters.", user_id)
            _finalize_run(client, run_id, status="success", **stats)
            tracker.save()
            return
        combined = pd.concat(filtered_parts, ignore_index=True)
        # URL-dedup across the merged set (a job could appear in both global and
        # local pulls). Keep first occurrence.
        if "job_url" in combined.columns:
            combined = combined.drop_duplicates(subset=["job_url"]).reset_index(drop=True)
        stats["filtered"] = int(len(combined))

        # 4. CV embedding pre-rank
        # All providers AUTO-DISCOVER their accounts (base → MULTI_ legacy →
        # _2/_3…), so adding an account is just a new secret — no code change.
        # core_llm round-robins across the comma-joined set and rate-limits each
        # account independently, so every account adds real headroom.
        cerebras_key = _discover_keys("CEREBRAS_API_KEY", "MULTI_CEREBRAS_API_KEY")
        groq_key = _discover_keys("GROQ_API_KEY", "MULTI_GROQ_API_KEY")
        # Gemini Flash Lite (lower-ranked verdicts) — rotated across accounts.
        gemini_key = _discover_keys("GEMINI_API_KEY")
        # Gemini embeddings (CV + job pre-rank) — its own account pool; the
        # embedding RPD is the capacity bottleneck, so a 2nd embed account here is
        # the highest-leverage scale-up. Falls back to the verdict keys if no
        # dedicated embed key is set.
        gemini_embed_key = _discover_keys("GEMINI_EMBED_API_KEY") or gemini_key

        combined, job_embeddings = attach_similarity(
            combined, user["cv_text"], gemini_embed_key,
            job_embed_cache=job_embed_cache,
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

            # W2: mint the tokenized feedback-page link for this (user, run).
            # APP_BASE_URL unset or token insert failing (e.g. migration 0012
            # not applied) degrades to the same email without a link — the
            # send itself must never be blocked by the feedback feature.
            feedback_url = None
            app_base = os.environ.get("APP_BASE_URL", "").strip().rstrip("/")
            if app_base:
                fb_token = create_feedback_token(user_id, run_id, client=client)
                if fb_token:
                    feedback_url = f"{app_base}/f/{fb_token}"
            else:
                logger.info(
                    "User %s: APP_BASE_URL not set — email goes out without a feedback link.",
                    user_id,
                )

            html = format_email_html(
                internships_df, jobs_df,
                {"scraped": stats["scraped"], "filtered": stats["filtered"], "approved": stats["approved"]},
                lower_ranked_df=lower_ranked,
                feedback_url=feedback_url,
            )
            ok, _info = send_email_transport(
                user["notification_email"],
                "Your Daily Job Alerts",
                html,
            )
            if not ok:
                logger.warning("User %s: email send failed — marking run success but email skipped.", user_id)
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

    finally:
        # Flush this user's LLM/embedding usage tally to Supabase, attributed to
        # them, and reset for the next user. Runs on success AND failure so even
        # a crashed run's spent calls are counted. Never raises into the loop.
        _flush_llm_usage(client, user_id)


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
    parser.add_argument("--manual", action="store_true",
                        help="Mark this run as a user-initiated manual dispatch "
                             "(stamps run_trigger='manual' and cancels today's "
                             "pending scheduled run). Use with --user-id.")
    parser.add_argument("--admin-override", action="store_true",
                        help="Admin forced run: bypass the 2/day budget cap. "
                             "Use with --user-id (ignored otherwise).")
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
    # Shared local-market jobs, collected once on first use this tick.
    local_cache = _LocalJobsCache() if INCLUDE_LOCAL_SOURCES else None
    # Shared job-embedding cache for THIS tick: a job seen by multiple due users
    # (the global scrape is shared) is embedded once, not per user. The embedding
    # is a pure function of the job text, so this is exact. Cleared each tick by
    # being a fresh dict — stale jobs never leak across ticks.
    job_embed_cache: dict = {}
    # A manual dispatch targets exactly one user (--user-id); guard against
    # --manual being passed for a whole-batch run, which would mis-stamp every
    # scheduled tick as manual.
    trigger = "manual" if (args.manual and args.user_id) else "scheduled"
    if args.manual and not args.user_id:
        logger.warning("--manual ignored: it requires --user-id. Treating as scheduled.")
    # Admin override only makes sense for a single targeted user; a batch override
    # would let the whole cohort blow past the budget, which is never intended.
    admin_override = bool(args.admin_override and args.user_id)
    if args.admin_override and not args.user_id:
        logger.warning("--admin-override ignored: it requires --user-id.")
    tick_start = time.time()
    for user in users:
        user_start = time.time()
        logger.info("--- Starting user %s (%s) ---", user["user_id"], trigger)
        _run_for_user(user, client, api_cache, dry_run=args.dry_run,
                      local_cache=local_cache, trigger=trigger,
                      admin_override=admin_override, job_embed_cache=job_embed_cache)
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
