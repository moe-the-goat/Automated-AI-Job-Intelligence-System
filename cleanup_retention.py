"""
RETENTION CLEANUP — keep Supabase within the free tier
------------------------------------------------------
Weekly housekeeping that bounds two unbounded-growth tables:

  * seen_jobs   — every evaluated URL per user, forever. Only the last
                  ~60 days are ever read (SupabaseJobTracker lookback), so
                  rows older than the retention window are dead weight.
                  Deleted outright.

  * job_results — one row per surfaced job per run. Old rows are purged
                  EXCEPT those a user bookmarked: a bookmark references
                  job_results.id, and bookmarked jobs must survive URL rot
                  indefinitely (they power Tab B). So we delete only
                  unbookmarked rows past the window.

Deleting a job_results row cascades to nothing load-bearing: feedback.job_result_id
is ON DELETE SET NULL (the append-only feedback row + its embedding survive, so
the RAG corpus is intact), and bookmarked rows are excluded up front so no
bookmark is ever orphaned.

Global, service-role, idempotent. Safe to re-run; --dry-run reports counts
without deleting.

CLI:
    python cleanup_retention.py --dry-run
    python cleanup_retention.py
    python cleanup_retention.py --seen-days 90 --results-days 120

Env:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_supabase import SupabaseConfigError, get_service_client

logger = get_logger(__name__)


SEEN_JOBS_RETENTION_DAYS = 90
JOB_RESULTS_RETENTION_DAYS = 90
# job_embeddings is only used for the 14-day semantic-dedup window, so it can be
# pruned much more aggressively than seen_jobs. A little slack (30d) keeps it
# robust to a missed weekly run without bloating the table.
JOB_EMBEDDINGS_RETENTION_DAYS = 30
_PAGE = 1000      # PostgREST default max rows per request
_DELETE_CHUNK = 100


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def partition_deletable(candidate_ids: list, bookmarked_ids: set) -> list:
    """From a page of old job_result ids, return those safe to delete —
    i.e. not referenced by any bookmark. Order preserved."""
    return [i for i in candidate_ids if i not in bookmarked_ids]


def chunks(seq: list, size: int = _DELETE_CHUNK):
    """Yield successive `size`-length slices of `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def cutoff_iso(days: int, *, now=None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now - timedelta(days=days)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _all_bookmarked_job_result_ids(client) -> set:
    """Every job_result_id referenced by a bookmark (keyset-paginated).

    Bookmarked jobs are preserved regardless of age, so we never delete these.
    """
    ids: set = set()
    last = 0
    while True:
        rows = (
            client.table("bookmarks")
            .select("id, job_result_id")
            .gt("id", last)
            .order("id")
            .limit(_PAGE)
            .execute()
        ).data or []
        if not rows:
            break
        for r in rows:
            if r.get("job_result_id") is not None:
                ids.add(r["job_result_id"])
        last = rows[-1]["id"]
        if len(rows) < _PAGE:
            break
    return ids


def purge_seen_jobs(client, days: int, *, dry_run: bool, now=None) -> int:
    """Delete seen_jobs older than `days`. Returns the count removed (or that
    would be removed in dry-run)."""
    cut = cutoff_iso(days, now=now)
    try:
        n = (
            client.table("seen_jobs")
            .select("job_url", count="exact", head=True)
            .lt("evaluated_at", cut)
            .execute()
        ).count or 0
    except Exception as e:
        logger.error("seen_jobs count failed: %s", e)
        return 0
    if n == 0:
        logger.info("seen_jobs: nothing older than %dd.", days)
        return 0
    if dry_run:
        logger.info("[dry-run] seen_jobs: would delete %d row(s) older than %dd.", n, days)
        return n
    try:
        client.table("seen_jobs").delete().lt("evaluated_at", cut).execute()
        logger.info("seen_jobs: deleted %d row(s) older than %dd.", n, days)
    except Exception as e:
        logger.error("seen_jobs delete failed: %s", e)
        return 0
    return n


def purge_by_timestamp(client, table: str, ts_column: str, days: int, *, dry_run: bool, now=None) -> int:
    """Delete rows from `table` whose `ts_column` is older than `days`.

    Generic single-condition purge for tables with no preserve-exceptions
    (unlike job_results, which must keep bookmarked rows). Used for
    job_embeddings. Returns the count removed (or that would be, in dry-run).
    """
    cut = cutoff_iso(days, now=now)
    try:
        n = (
            client.table(table)
            .select(ts_column, count="exact", head=True)
            .lt(ts_column, cut)
            .execute()
        ).count or 0
    except Exception as e:
        logger.error("%s count failed: %s", table, e)
        return 0
    if n == 0:
        logger.info("%s: nothing older than %dd.", table, days)
        return 0
    if dry_run:
        logger.info("[dry-run] %s: would delete %d row(s) older than %dd.", table, n, days)
        return n
    try:
        client.table(table).delete().lt(ts_column, cut).execute()
        logger.info("%s: deleted %d row(s) older than %dd.", table, n, days)
    except Exception as e:
        logger.error("%s delete failed: %s", table, e)
        return 0
    return n


def purge_job_results(client, days: int, *, dry_run: bool, now=None) -> int:
    """Delete job_results older than `days` that no bookmark references.

    Keyset-paginates by id so deleting within a page never shifts later pages,
    and bookmarked rows (left in place) are simply skipped past.
    """
    cut = cutoff_iso(days, now=now)
    bookmarked = _all_bookmarked_job_result_ids(client)
    total = 0
    last = 0
    while True:
        try:
            rows = (
                client.table("job_results")
                .select("id")
                .lt("created_at", cut)
                .gt("id", last)
                .order("id")
                .limit(_PAGE)
                .execute()
            ).data or []
        except Exception as e:
            logger.error("job_results fetch failed: %s", e)
            break
        if not rows:
            break
        last = rows[-1]["id"]
        deletable = partition_deletable([r["id"] for r in rows], bookmarked)
        total += len(deletable)
        if not dry_run and deletable:
            for chunk in chunks(deletable):
                try:
                    client.table("job_results").delete().in_("id", chunk).execute()
                except Exception as e:
                    logger.error("job_results delete chunk failed: %s", e)
        if len(rows) < _PAGE:
            break

    verb = "would delete" if dry_run else "deleted"
    logger.info("job_results: %s %d unbookmarked row(s) older than %dd (%d bookmarked rows preserved).",
                verb, total, days, len(bookmarked))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Retention cleanup for seen_jobs, job_results, job_embeddings.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts; delete nothing.")
    parser.add_argument("--seen-days", type=int, default=SEEN_JOBS_RETENTION_DAYS)
    parser.add_argument("--results-days", type=int, default=JOB_RESULTS_RETENTION_DAYS)
    parser.add_argument("--embeddings-days", type=int, default=JOB_EMBEDDINGS_RETENTION_DAYS)
    args = parser.parse_args(argv)

    configure_logging()

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    logger.info("Retention cleanup%s — seen_jobs>%dd, job_results>%dd, job_embeddings>%dd.",
                " [dry-run]" if args.dry_run else "",
                args.seen_days, args.results_days, args.embeddings_days)
    seen = purge_seen_jobs(client, args.seen_days, dry_run=args.dry_run)
    results = purge_job_results(client, args.results_days, dry_run=args.dry_run)
    embeds = purge_by_timestamp(
        client, "job_embeddings", "embedded_at", args.embeddings_days, dry_run=args.dry_run
    )
    logger.info("Cleanup %s: seen_jobs=%d, job_results=%d, job_embeddings=%d.",
                "preview" if args.dry_run else "done", seen, results, embeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
