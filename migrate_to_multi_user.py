"""
DATA MIGRATION — single-user corpus → Supabase (B9a)
----------------------------------------------------
One-time, idempotent port of Mohammad's existing single-user data into the
multi-user Supabase schema, attached to his web-app account (user #1).

Sources:
  * Private logs repo (via the GitHub Contents API, same as core_feedback):
      - data/feedback_log.json        → public.feedback
      - data/feedback_embeddings.json → public.feedback_embeddings  (positionally aligned)
      - data/candidate_preferences.txt → preferences.candidate_preferences
  * Local working tree:
      - data/reputation.json          → public.reputation

Design decisions that matter for a one-time migration:

  1. MULTISET idempotency. The feedback table is append-only with a bigserial
     PK — there's no natural unique key. Naive "skip if (url, type) exists"
     dedup would collapse genuine repeat reactions (e.g. the same job
     bookmarked then later applied are different types and survive, but two
     identical reactions are real history too). We compare COUNTS per key:
     the DB should end up with max(already-there, in-log) rows for each key.
     Re-running inserts nothing; a half-finished first run resumes exactly
     where it stopped.

  2. Precise embedding pairing. A feedback row and its pre-computed embedding
     must stay matched — a mispaired vector silently corrupts that user's RAG
     signal. We insert one feedback row, capture its real id, then insert that
     row's embedding. Slower than a bulk insert, but for a ~dozens-of-rows
     one-time port, correctness wins over speed. Rows whose stored embedding
     is null/missing are inserted WITHOUT an embedding — the runner's
     ensure_feedback_embeddings() backfills them on the next tick.

  3. Migration-independence. We reconcile profiles.feedback_count at the end
     with an absolute SET (not +=). That's correct whether or not migration
     0006's insert-trigger is applied, and idempotent on re-run either way.

CLI:
    python migrate_to_multi_user.py --email you@example.com
    python migrate_to_multi_user.py --user-id <uuid>
    python migrate_to_multi_user.py --email you@example.com --dry-run
    python migrate_to_multi_user.py --user-id <uuid> --skip-reputation

Env:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   (the worker's service-role creds)
    LOGS_REPO, LOGS_REPO_TOKEN                 (the private logs repo)
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_feedback import (
    LOG_PATH,
    EMBEDDINGS_PATH,
    PREFERENCES_PATH,
    LOCAL_REPUTATION_PATH,
    LogsRepoAuthError,
    _read_file,
    verify_logs_repo_access,
)
from pipeline.core_supabase import SupabaseConfigError, get_service_client
# Canonical pgvector-literal formatter lives in core_feedback_supabase; re-export
# under the historical name so this module (and its tests) keep one shared impl.
from pipeline.core_feedback_supabase import to_pgvector_literal as vector_literal

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (no I/O — unit-tested directly)
# ─────────────────────────────────────────────────────────────────────────────

# Maps the reputation.json list keys to the reputation table's pattern_type enum.
REPUTATION_TYPE_MAP = {
    "blacklist_name_patterns": "blacklist_name",
    "blacklist_handle_patterns": "blacklist_handle",
    "trust_boost": "trust_boost",
}


def _feedback_key(entry: dict) -> tuple:
    """Stable natural key for multiset dedup: (job_url, feedback_type, note).

    Accepts either a logs-repo entry (uses `feedback`) or a Supabase row
    (uses `feedback_type`) so the same function keys both sides of the
    comparison. Note is normalized to '' so null/empty compare equal.
    """
    url = str(entry.get("job_url", "")).strip()
    ftype = str(entry.get("feedback_type") or entry.get("feedback") or "").strip().lower()
    note = (entry.get("note") or "").strip()
    return (url, ftype, note)


def plan_inserts(log_entries: list, existing_rows: list) -> list:
    """Return the indices of `log_entries` that still need inserting.

    Multiset semantics: if the DB already holds N rows for a key and the log
    has M entries for that key, we insert the last max(M-N, 0) of them, in
    log order. This makes the migration idempotent AND faithful to genuine
    repeat reactions.
    """
    have = Counter(_feedback_key(r) for r in existing_rows)
    to_insert = []
    for i, entry in enumerate(log_entries):
        key = _feedback_key(entry)
        if have.get(key, 0) > 0:
            have[key] -= 1          # this log entry is already represented in the DB
            continue
        to_insert.append(i)
    return to_insert


def reputation_rows(rep: dict, added_by: Optional[str]) -> list:
    """Flatten reputation.json into rows for the reputation table.

    Dedups within the file (case-insensitive, by (type, pattern)) so a
    hand-edited list with accidental repeats doesn't fight the table's
    composite PK. `added_by` is the migrating user's uuid (nullable).
    """
    if not isinstance(rep, dict):
        return []
    out = []
    seen = set()
    for list_key, pattern_type in REPUTATION_TYPE_MAP.items():
        for raw in rep.get(list_key, []) or []:
            pattern = str(raw).strip().lower()
            if not pattern:
                continue
            dedup_key = (pattern_type, pattern)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            row = {"pattern_type": pattern_type, "pattern": pattern}
            if added_by:
                row["added_by"] = added_by
            out.append(row)
    return out


def parse_submitted_at(date_str) -> Optional[str]:
    """Best-effort ISO-8601 parse so original feedback timestamps survive.

    Returns a normalized ISO string on success, or None to let the DB default
    (now()) apply. Never raises — an unparseable date just falls back.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None
    # Tolerate a trailing Z (datetime.fromisoformat rejects it before 3.11).
    candidate = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def feedback_insert_row(entry: dict, user_id: str) -> dict:
    """Build the public.feedback insert payload from a logs-repo entry.

    job_result_id is NULL — migrated history predates the multi-user
    job_results table, so there's no run row to link to. The feedback table
    allows null job_result_id by design (on delete set null).
    """
    row = {
        "user_id": user_id,
        "job_result_id": None,
        "job_url": str(entry.get("job_url", "")).strip()[:500],
        "title": (str(entry.get("title", "")).strip() or None),
        "company": (str(entry.get("company", "")).strip() or None),
        "feedback_type": str(entry.get("feedback") or entry.get("feedback_type") or "").strip().lower(),
        "note": (str(entry.get("note", "")).strip() or None),
    }
    submitted = parse_submitted_at(entry.get("date"))
    if submitted:
        row["submitted_at"] = submitted
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Source loading (I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _load_logs_repo_json(repo, token, path) -> Optional[dict]:
    try:
        text, _ = _read_file(repo, path, token)
    except LogsRepoAuthError as e:
        logger.critical("Cannot read %s: %s", path, e)
        return None
    if not text:
        logger.info("%s is empty or missing.", path)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error("%s is malformed JSON.", path)
        return None


def _load_local_reputation() -> Optional[dict]:
    try:
        with open(LOCAL_REPUTATION_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("reputation.json unreadable (%s) — skipping reputation migration.", e)
        return None


def resolve_user_id(client, *, user_id: Optional[str], email: Optional[str]) -> Optional[str]:
    """Return the target user_id, looking it up by email via the auth admin API
    when only an email was given. --user-id always wins if provided.
    """
    if user_id:
        return user_id
    if not email:
        return None
    email_lc = email.strip().lower()
    try:
        users = client.auth.admin.list_users()
    except Exception as e:
        logger.critical("Could not list auth users to resolve email %s: %s", email, e)
        return None
    # supabase-py has returned either a bare list or an object with .users
    # across 2.x minor versions — tolerate both.
    if hasattr(users, "users"):
        users = users.users
    for u in users or []:
        u_email = getattr(u, "email", None) or (u.get("email") if isinstance(u, dict) else None)
        u_id = getattr(u, "id", None) or (u.get("id") if isinstance(u, dict) else None)
        if u_email and u_email.strip().lower() == email_lc:
            return u_id
    logger.critical("No auth user found with email %s. Has the account signed up yet?", email)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Migration steps (I/O)
# ─────────────────────────────────────────────────────────────────────────────

def migrate_feedback(client, user_id, log_data, embed_data, *, dry_run) -> dict:
    """Insert feedback rows + paired embeddings. Returns a stats dict."""
    log_entries = (log_data or {}).get("entries", []) if isinstance(log_data, dict) else []
    embed_entries = (embed_data or {}).get("entries", []) if isinstance(embed_data, dict) else []

    stats = {"in_log": len(log_entries), "inserted": 0, "skipped": 0,
             "embeddings": 0, "embeds_deferred": 0, "failed": 0}
    if not log_entries:
        logger.info("Feedback: nothing in the log to migrate.")
        return stats

    if len(embed_entries) != len(log_entries):
        logger.warning(
            "Feedback: log has %d entries but embeddings file has %d. "
            "Pairing by index up to the shorter; unpaired rows get backfilled by the runner.",
            len(log_entries), len(embed_entries),
        )

    # Existing rows for multiset dedup.
    existing = _fetch_existing_feedback(client, user_id)
    indices = plan_inserts(log_entries, existing)
    stats["skipped"] = len(log_entries) - len(indices)
    logger.info("Feedback: %d already present, %d to insert.", stats["skipped"], len(indices))

    if dry_run:
        logger.info("[dry-run] Would insert %d feedback row(s).", len(indices))
        return stats

    for i in indices:
        entry = log_entries[i]
        row = feedback_insert_row(entry, user_id)
        if not row["job_url"] or not row["feedback_type"]:
            logger.warning("Feedback[%d]: missing url/type — skipping.", i)
            stats["failed"] += 1
            continue
        try:
            # insert() returns the inserted row(s) as a list (returning=representation);
            # there is no .select()/.single() on an insert builder.
            resp = client.table("feedback").insert(row).execute()
            new_id = resp.data[0]["id"] if resp.data else None
        except Exception as e:
            logger.error("Feedback[%d] insert failed: %s", i, e)
            stats["failed"] += 1
            continue
        if new_id is None:
            logger.error("Feedback[%d] insert returned no id — skipping embedding.", i)
            stats["failed"] += 1
            continue
        stats["inserted"] += 1

        vec = embed_entries[i].get("embedding") if i < len(embed_entries) and isinstance(embed_entries[i], dict) else None
        literal = vector_literal(vec)
        if literal is None:
            stats["embeds_deferred"] += 1
            continue
        try:
            client.table("feedback_embeddings").upsert(
                {"feedback_id": new_id, "user_id": user_id, "embedding": literal},
                on_conflict="feedback_id",
            ).execute()
            stats["embeddings"] += 1
        except Exception as e:
            logger.error("Feedback[%d] embedding insert failed (will backfill later): %s", i, e)
            stats["embeds_deferred"] += 1

    return stats


def _fetch_existing_feedback(client, user_id) -> list:
    try:
        resp = (
            client.table("feedback")
            .select("job_url, feedback_type, note")
            .eq("user_id", user_id)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        logger.error("Could not read existing feedback for %s: %s", user_id, e)
        return []


def migrate_candidate_preferences(client, user_id, repo, token, *, dry_run) -> bool:
    try:
        text, _ = _read_file(repo, PREFERENCES_PATH, token)
    except LogsRepoAuthError as e:
        logger.critical("Cannot read candidate preferences: %s", e)
        return False
    text = (text or "").strip()
    if not text:
        logger.info("Candidate preferences: empty — nothing to migrate.")
        return True
    if dry_run:
        logger.info("[dry-run] Would set preferences.candidate_preferences (%d chars).", len(text))
        return True
    try:
        client.table("preferences").update(
            {"candidate_preferences": text}
        ).eq("user_id", user_id).execute()
        logger.info("Candidate preferences: migrated (%d chars).", len(text))
        return True
    except Exception as e:
        logger.error("Candidate preferences update failed: %s", e)
        return False


def migrate_reputation(client, user_id, *, dry_run) -> dict:
    rep = _load_local_reputation()
    rows = reputation_rows(rep, user_id) if rep else []
    stats = {"rows": len(rows), "upserted": 0}
    if not rows:
        return stats
    if dry_run:
        logger.info("[dry-run] Would upsert %d reputation pattern(s).", len(rows))
        return stats
    try:
        # Composite PK (pattern_type, pattern) makes this idempotent.
        client.table("reputation").upsert(
            rows, on_conflict="pattern_type,pattern", ignore_duplicates=True
        ).execute()
        stats["upserted"] = len(rows)
        logger.info("Reputation: upserted %d pattern(s).", len(rows))
    except Exception as e:
        logger.error("Reputation upsert failed: %s", e)
    return stats


def reconcile_feedback_count(client, user_id, *, dry_run) -> Optional[int]:
    """Absolute SET of profiles.feedback_count to the true row count.

    Correct with OR without migration 0006's trigger, and idempotent either
    way. Returns the reconciled count (or None on failure / dry-run).
    """
    try:
        resp = (
            client.table("feedback")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        true_count = resp.count if resp.count is not None else len(resp.data or [])
    except Exception as e:
        logger.error("Could not count feedback for reconcile: %s", e)
        return None
    if dry_run:
        logger.info("[dry-run] Would set profiles.feedback_count = %d.", true_count)
        return true_count
    try:
        client.table("profiles").update(
            {"feedback_count": true_count}
        ).eq("user_id", user_id).execute()
        logger.info("Reconciled profiles.feedback_count = %d.", true_count)
        return true_count
    except Exception as e:
        logger.error("feedback_count reconcile update failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Port single-user data into Supabase (B9a).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user-id", help="Target user UUID.")
    target.add_argument("--email", help="Target user email (looked up via auth admin).")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    parser.add_argument("--skip-feedback", action="store_true", help="Skip feedback + embeddings.")
    parser.add_argument("--skip-preferences", action="store_true", help="Skip candidate_preferences.")
    parser.add_argument("--skip-reputation", action="store_true", help="Skip reputation patterns.")
    args = parser.parse_args(argv)

    configure_logging()

    import os
    repo = os.environ.get("LOGS_REPO")
    token = os.environ.get("LOGS_REPO_TOKEN")
    if not repo or not token:
        logger.critical("LOGS_REPO and LOGS_REPO_TOKEN are required to read the source corpus.")
        return 2
    if not verify_logs_repo_access(repo, token):
        logger.critical("Logs repo unreachable — aborting (see CRITICAL log above).")
        return 2

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    user_id = resolve_user_id(client, user_id=args.user_id, email=args.email)
    if not user_id:
        return 2
    logger.info("Migrating into user_id=%s%s.", user_id, " [dry-run]" if args.dry_run else "")

    if not args.skip_feedback:
        log_data = _load_logs_repo_json(repo, token, LOG_PATH)
        embed_data = _load_logs_repo_json(repo, token, EMBEDDINGS_PATH)
        fb_stats = migrate_feedback(client, user_id, log_data, embed_data, dry_run=args.dry_run)
        logger.info("Feedback summary: %s", fb_stats)

    if not args.skip_preferences:
        migrate_candidate_preferences(client, user_id, repo, token, dry_run=args.dry_run)

    if not args.skip_reputation:
        rep_stats = migrate_reputation(client, user_id, dry_run=args.dry_run)
        logger.info("Reputation summary: %s", rep_stats)

    if not args.skip_feedback:
        reconcile_feedback_count(client, user_id, dry_run=args.dry_run)

    logger.info("Migration %s.", "preview complete" if args.dry_run else "complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
