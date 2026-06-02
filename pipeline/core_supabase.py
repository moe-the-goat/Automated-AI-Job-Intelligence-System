"""
CORE SUPABASE MODULE
--------------------
Thin server-side wrapper around supabase-py for the multi-user worker.

The worker always uses the **service-role key**, which bypasses Row-Level
Security by design. RLS is enforced for the Next.js app (anon key); the
worker is trusted infrastructure and reads/writes any user's rows.

Also provides `SupabaseJobTracker` — a drop-in replacement for
`pipeline.core_filter.JobTracker` that persists seen URLs in the per-user
`seen_jobs` table instead of `seen_jobs.json`. Same interface (`is_seen`,
`mark_seen`, `save`) so `apply_pipeline_filters(df, tracker=...)` doesn't
care which one it gets.

Env:
  SUPABASE_URL                 required
  SUPABASE_SERVICE_ROLE_KEY    required — server-side, NEVER ship to the browser
"""

import os
from typing import Optional

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


# Cache the client at module level so a multi-user run that touches Supabase
# 50 times reuses the same connection pool instead of reconnecting per call.
_CACHED_CLIENT = None


class SupabaseConfigError(RuntimeError):
    """Raised at worker boot when SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are missing.

    Fail loud at startup rather than silently producing zero-user runs.
    """


def get_service_client():
    """Return a cached service-role Supabase client.

    Lazy import keeps `supabase` off the import path for the single-user
    `scraper.py` / `local_companies.py` flows that don't touch Supabase.
    """
    global _CACHED_CLIENT
    if _CACHED_CLIENT is not None:
        return _CACHED_CLIENT

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must both be set "
            "for the multi-user worker."
        )

    from supabase import create_client  # lazy: not needed in single-user pipeline

    _CACHED_CLIENT = create_client(url, key)
    return _CACHED_CLIENT


def reset_client_cache():
    """Test hook — drop the cached client so the next get_service_client() reconnects."""
    global _CACHED_CLIENT
    _CACHED_CLIENT = None


# ---------------------------------------------------------------------------
# SupabaseJobTracker
# ---------------------------------------------------------------------------

# Cap how many seen_jobs we load per user. The table is unbounded over time,
# but the only useful contents are URLs from runs within the lookback window
# of the underlying job sources (LinkedIn/Indeed/APIs all surface at most ~30
# days of history). Reading 2 years of stale rows just to dedup against
# yesterday's batch wastes round-trip bytes.
SEEN_JOBS_LOOKBACK_DAYS = 60


class SupabaseJobTracker:
    """Drop-in replacement for core_filter.JobTracker, backed by `seen_jobs`.

    Interface contract (must match JobTracker):
      - is_seen(url) -> bool
      - mark_seen(url) -> None
      - save() -> None

    Loads the user's recent seen URLs into memory on construction (one SELECT)
    so every is_seen() during the run is O(1). New URLs are buffered and
    flushed in a single UPSERT on save().

    Per-user isolation is enforced at the query level — the worker uses
    service-role (bypasses RLS), so we MUST pass user_id explicitly on every
    read AND write or we'd risk cross-user leaks.
    """

    def __init__(self, user_id: str, client=None):
        if not user_id:
            raise ValueError("SupabaseJobTracker requires a user_id")
        self.user_id = user_id
        self._client = client or get_service_client()
        self.seen_urls: set = set()
        self._pending: list[str] = []
        self.load()

    def load(self):
        try:
            from datetime import datetime, timedelta, timezone

            since = (
                datetime.now(timezone.utc) - timedelta(days=SEEN_JOBS_LOOKBACK_DAYS)
            ).isoformat()
            resp = (
                self._client.table("seen_jobs")
                .select("job_url")
                .eq("user_id", self.user_id)
                .gte("evaluated_at", since)
                .execute()
            )
            self.seen_urls = {
                row["job_url"] for row in (resp.data or []) if row.get("job_url")
            }
            logger.info(
                "SupabaseJobTracker: loaded %d seen URL(s) for user %s.",
                len(self.seen_urls), self.user_id,
            )
        except Exception as e:
            logger.warning(
                "SupabaseJobTracker.load failed for user %s: %s. Treating as empty.",
                self.user_id, e,
            )
            self.seen_urls = set()

    def is_seen(self, url: Optional[str]) -> bool:
        return bool(url) and url in self.seen_urls

    def mark_seen(self, url: Optional[str]) -> None:
        if not url:
            return
        if url not in self.seen_urls:
            self.seen_urls.add(url)
            self._pending.append(url)

    def save(self) -> None:
        """Flush newly-seen URLs to the seen_jobs table in a single UPSERT."""
        if not self._pending:
            return
        rows = [{"user_id": self.user_id, "job_url": url} for url in self._pending]
        try:
            # on_conflict="user_id,job_url" matches the composite PK so re-marks
            # are a no-op (preserves the original evaluated_at).
            self._client.table("seen_jobs").upsert(
                rows, on_conflict="user_id,job_url", ignore_duplicates=True
            ).execute()
            logger.info(
                "SupabaseJobTracker.save: persisted %d new URL(s) for user %s.",
                len(rows), self.user_id,
            )
            self._pending = []
        except Exception as e:
            # Don't lose the buffer — the cron tick will retry next run.
            logger.error(
                "SupabaseJobTracker.save FAILED for user %s (%d rows queued): %s",
                self.user_id, len(rows), e,
            )


# ---------------------------------------------------------------------------
# Per-user job embedding history (semantic dedup — multi-user analog of
# core_embedding's local embedding_history.json)
# ---------------------------------------------------------------------------

# Only compare against recent history — mirrors the legacy 14-day rolling cache.
# Older reposts are no longer "the same job reposted", just normal churn.
JOB_EMBEDDING_LOOKBACK_DAYS = 14


def load_job_embedding_history(user_id: str, client=None) -> dict:
    """Return {job_url: embedding_vector} of this user's recently-surfaced jobs.

    Multi-user equivalent of core_embedding.load_embedding_history(): the rolling
    window of prerank embeddings that drop_semantic_duplicates() compares new
    jobs against. Reads only the last JOB_EMBEDDING_LOOKBACK_DAYS. Returns {} on
    any error so dedup degrades to a no-op rather than crashing the run.
    """
    if not user_id:
        return {}
    client = client or get_service_client()
    from datetime import datetime, timedelta, timezone

    since = (
        datetime.now(timezone.utc) - timedelta(days=JOB_EMBEDDING_LOOKBACK_DAYS)
    ).isoformat()
    try:
        resp = (
            client.table("job_embeddings")
            .select("job_url, embedding")
            .eq("user_id", user_id)
            .gte("embedded_at", since)
            .execute()
        )
    except Exception as e:
        logger.warning("load_job_embedding_history failed for %s: %s", user_id, e)
        return {}

    out = {}
    for row in resp.data or []:
        url = row.get("job_url")
        vec = row.get("embedding")
        if url and isinstance(vec, list) and vec:
            out[url] = vec
    return out


def save_job_embedding_history(user_id: str, embeddings: dict, client=None) -> int:
    """Upsert {job_url: embedding_vector} for this user. Returns rows written.

    Called with the embeddings of jobs that survived to delivery, so next run's
    dedup has fresh history. embedded_at refreshes on conflict so a re-surfaced
    job stays inside the rolling window. Stored as a plain jsonb array (the
    comparison is a Python cosine loop, not a pgvector query). Best-effort.
    """
    if not user_id or not embeddings:
        return 0
    client = client or get_service_client()
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        {"user_id": user_id, "job_url": url, "embedding": list(vec), "embedded_at": now_iso}
        for url, vec in embeddings.items()
        if url and vec is not None
    ]
    if not rows:
        return 0
    try:
        # Chunk to keep the payload sane — vectors are large.
        written = 0
        for i in range(0, len(rows), 100):
            client.table("job_embeddings").upsert(
                rows[i:i + 100], on_conflict="user_id,job_url"
            ).execute()
            written += len(rows[i:i + 100])
        logger.info("save_job_embedding_history: wrote %d embedding(s) for user %s.", written, user_id)
        return written
    except Exception as e:
        logger.error("save_job_embedding_history failed for %s: %s", user_id, e)
        return 0
