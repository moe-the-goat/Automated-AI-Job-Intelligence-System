"""
FEEDBACK DIGEST (multi-user) — B7c
----------------------------------
Per-user analog of `feedback_digest.py`. Runs on its own cron tick (every
5 days). For each user whose digest is stale AND whose feedback count is
still under the RAG threshold, summarizes their feedback log into a 5-8
sentence candidate preference profile and writes it to
`preferences.candidate_preferences`.

A user is "due for a digest" when:
  * `preferences.is_active` is true, AND
  * `preferences.last_digest_at` is null OR older than DIGEST_INTERVAL_DAYS, AND
  * the user's feedback count is below RAG_FEEDBACK_THRESHOLD.

Users at/above the threshold are skipped automatically — the runner switches
them to per-job retrieval, so a refreshed global summary would never be read.
This mirrors the early-exit in the single-user `feedback_digest.py`.

CLI:
    python feedback_digest_multi_user.py
    python feedback_digest_multi_user.py --user-id <uuid>
    python feedback_digest_multi_user.py --force         # ignore the interval gate
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_supabase import SupabaseConfigError, get_service_client
from pipeline.core_feedback_supabase import (
    RAG_FEEDBACK_THRESHOLD,
    count_feedback_entries,
    format_entry_text,
)
# Reuse the prompt verbatim so single-user and multi-user digests stay
# semantically aligned — same instruction → comparable preference profiles.
from feedback_digest import SUMMARY_PROMPT

logger = get_logger(__name__)


DIGEST_INTERVAL_DAYS = 5


def _load_due_user_ids(client, *, only_user_id: Optional[str], force: bool) -> list[str]:
    """Return user_ids whose digest is due (or all whitelisted users if --force)."""
    cutoff_iso = (
        datetime.now(timezone.utc) - timedelta(days=DIGEST_INTERVAL_DAYS)
    ).isoformat()

    query = client.table("preferences").select("user_id, last_digest_at").eq("is_active", True)

    if only_user_id:
        query = query.eq("user_id", only_user_id)

    try:
        resp = query.execute()
    except Exception as e:
        logger.critical("Failed to load preferences for digest: %s", e)
        return []

    user_ids = []
    for row in resp.data or []:
        uid = row["user_id"]
        if force:
            user_ids.append(uid)
            continue
        last = row.get("last_digest_at")
        if last is None or last < cutoff_iso:
            user_ids.append(uid)
    return user_ids


def _fetch_user_feedback(client, user_id: str) -> list[dict]:
    """Return the user's feedback rows in chronological order."""
    try:
        resp = (
            client.table("feedback")
            .select("feedback_type, title, company, note, submitted_at")
            .eq("user_id", user_id)
            .order("submitted_at", desc=False)
            .execute()
        )
    except Exception as e:
        logger.error("Digest: feedback fetch failed for %s: %s", user_id, e)
        return []
    return resp.data or []


def _summarize(entries: list[dict], *, cerebras_key: str, groq_key: str) -> Optional[str]:
    """Call Cerebras with Groq fallback. Returns None on empty/failed output."""
    formatted = "\n".join(format_entry_text(e) for e in entries)
    prompt = SUMMARY_PROMPT.format(entries=formatted)

    from pipeline.core_llm import call_llm_with_fallback  # lazy

    try:
        summary = call_llm_with_fallback(
            prompt,
            cerebras_key=cerebras_key,
            groq_key=groq_key,
            max_attempts=4,
            label="feedback-digest-multiuser",
        )
    except Exception as e:
        logger.error("Digest LLM call failed: %s", e)
        return None
    summary = (summary or "").strip()
    return summary or None


def _persist_summary(client, user_id: str, summary: str) -> bool:
    try:
        client.table("preferences").update({
            "candidate_preferences": summary,
            "last_digest_at": datetime.now(timezone.utc).isoformat(),
        }).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        logger.error("Digest persist failed for %s: %s", user_id, e)
        return False


def _bump_digest_timestamp_only(client, user_id: str):
    """No summary to refresh, but record that we tried so we don't re-pick this
    user on the next tick (we still respect the RAG threshold for the actual gate).
    """
    try:
        client.table("preferences").update(
            {"last_digest_at": datetime.now(timezone.utc).isoformat()}
        ).eq("user_id", user_id).execute()
    except Exception as e:
        logger.warning("Digest timestamp bump failed for %s: %s", user_id, e)


def run_for_user(client, user_id: str, *, cerebras_key: str, groq_key: str) -> bool:
    """Returns True if the cycle for this user finished cleanly (including
    skip cases like 'no entries' or 'above RAG threshold'). False on hard
    failure that should be surfaced in CI.
    """
    total = count_feedback_entries(user_id, client=client)
    if total == 0:
        logger.info("Digest: user %s has no feedback — skipping.", user_id)
        _bump_digest_timestamp_only(client, user_id)
        return True
    if total >= RAG_FEEDBACK_THRESHOLD:
        logger.info(
            "Digest: user %s has %d entries >= RAG threshold %d. Skipping summary "
            "(runner is using per-job retrieval).",
            user_id, total, RAG_FEEDBACK_THRESHOLD,
        )
        _bump_digest_timestamp_only(client, user_id)
        return True

    entries = _fetch_user_feedback(client, user_id)
    if not entries:
        logger.warning(
            "Digest: profiles.feedback_count says %d for %s but feedback table returned 0. "
            "Counter likely drifted — skipping.",
            total, user_id,
        )
        return True

    summary = _summarize(entries, cerebras_key=cerebras_key, groq_key=groq_key)
    if not summary:
        logger.error("Digest: empty/failed summary for %s — preferences NOT updated.", user_id)
        return False

    if _persist_summary(client, user_id, summary):
        logger.info(
            "Digest: user %s — summarized %d entries (%d chars).",
            user_id, len(entries), len(summary),
        )
        return True
    return False


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-user feedback digest (B7c).")
    parser.add_argument("--user-id", default=None,
                        help="Process only this user (UUID).")
    parser.add_argument("--force", action="store_true",
                        help="Ignore the 5-day interval gate (digest every active user).")
    args = parser.parse_args(argv)

    configure_logging()

    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not cerebras_key and not groq_key:
        logger.critical("CEREBRAS_API_KEY or GROQ_API_KEY required for summarization.")
        return 2

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    user_ids = _load_due_user_ids(client, only_user_id=args.user_id, force=args.force)
    logger.info("Digest: %d user(s) due.", len(user_ids))

    tick_start = time.time()
    failures = 0
    for uid in user_ids:
        try:
            ok = run_for_user(client, uid, cerebras_key=cerebras_key, groq_key=groq_key)
            if not ok:
                failures += 1
        except Exception as e:
            failures += 1
            logger.error("Digest: unexpected error for %s: %s", uid, e)

    logger.info(
        "Digest tick complete: %d processed, %d failure(s), %.1fs.",
        len(user_ids), failures, time.time() - tick_start,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
