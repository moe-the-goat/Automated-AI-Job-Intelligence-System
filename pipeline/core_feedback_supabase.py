"""
CORE FEEDBACK (Supabase) MODULE
-------------------------------
Supabase-backed analog of `pipeline.core_feedback`. Same public API shape so
multi_user_runner.py can plug it into the same RAG / digest switch the
single-user pipeline uses.

Mapping vs the GitHub Contents API version:

  load_candidate_preferences(user_id)   -> SELECT preferences.candidate_preferences
  count_feedback_entries(user_id)       -> SELECT profiles.feedback_count  (denormalized)
  load_feedback_embeddings(user_id)     -> SELECT feedback + feedback_embeddings
                                            shaped as {"entries": [{"text", "embedding"}]}
                                            so core_embedding.retrieve_relevant_feedback
                                            consumes it unchanged.
  ensure_feedback_embeddings(user_id, embed_api_key)
                                        -> embed any feedback rows with no embedding
                                            yet, insert into feedback_embeddings,
                                            return total feedback count.

ingest_pending_feedback() has no analog — in the multi-user world feedback
lands directly in the `feedback` table via the Next.js /api/feedback route,
so there's nothing to drain at run start.

The RAG_FEEDBACK_THRESHOLD / RAG_TOP_K constants are re-exported from the
old module so both paths stay in lockstep when we tune them.
"""

from typing import Optional

from pipeline.logging_setup import get_logger
from pipeline.core_feedback import RAG_FEEDBACK_THRESHOLD, RAG_TOP_K
from pipeline.core_supabase import get_service_client

logger = get_logger(__name__)

__all__ = [
    "RAG_FEEDBACK_THRESHOLD",
    "RAG_TOP_K",
    "load_candidate_preferences",
    "count_feedback_entries",
    "load_feedback_embeddings",
    "ensure_feedback_embeddings",
    "format_entry_text",
]


def format_entry_text(entry: dict) -> str:
    """Canonical short-text representation of a Supabase feedback row.

    Matches `core_feedback.format_entry_text` for the GitHub-backed log so
    embeddings created here and there are directly comparable. The schema
    drift is small (Supabase row has `feedback_type` not `feedback`,
    `company`/`title` are denormalized columns rather than nested) — we
    paper over it here, in one place.
    """
    if not isinstance(entry, dict):
        return ""
    feedback_type = entry.get("feedback_type") or entry.get("feedback") or "unknown"
    title = entry.get("title") or "?"
    company = entry.get("company") or "?"
    parts = [f"[{feedback_type}]", f"{title} @ {company}"]
    note = entry.get("note")
    if note:
        parts.append(f"— note: {note}")
    return " ".join(parts)


def load_candidate_preferences(user_id: str, client=None) -> str:
    """Return preferences.candidate_preferences for the user, or '' on miss/error."""
    if not user_id:
        return ""
    client = client or get_service_client()
    try:
        resp = (
            client.table("preferences")
            .select("candidate_preferences")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return ""
        return rows[0].get("candidate_preferences") or ""
    except Exception as e:
        logger.warning("load_candidate_preferences failed for %s: %s", user_id, e)
        return ""


def count_feedback_entries(user_id: str, client=None) -> int:
    """Return profiles.feedback_count for the user.

    Denormalized counter is incremented by the /api/feedback route on insert
    so a fast SELECT replaces a COUNT(*) over the feedback table on every run.
    Falls back to 0 silently — a wrong-low count just means we stay in digest
    mode for one more cycle, never a silent crash.
    """
    if not user_id:
        return 0
    client = client or get_service_client()
    try:
        resp = (
            client.table("profiles")
            .select("feedback_count")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            return 0
        return int(rows[0].get("feedback_count") or 0)
    except Exception as e:
        logger.warning("count_feedback_entries failed for %s: %s", user_id, e)
        return 0


def load_feedback_embeddings(user_id: str, client=None) -> dict:
    """Return the user's feedback corpus shaped for core_embedding.retrieve_relevant_feedback.

    Shape: {"entries": [{"text": str, "embedding": list[float]}, ...]}

    Joins `feedback` with `feedback_embeddings` on feedback.id. Rows without
    an embedding yet (ensure_feedback_embeddings hasn't caught up) are
    dropped — retrieval would skip them anyway and including them just
    inflates the result list.
    """
    empty = {"entries": []}
    if not user_id:
        return empty
    client = client or get_service_client()

    try:
        # Inner join via Supabase's foreign-key embed syntax. `embedding` is a
        # pgvector — supabase-py returns it as a string like "[0.1,0.2,...]"
        # on the wire (PostgREST serializes vectors that way), so we parse
        # below.
        resp = (
            client.table("feedback")
            .select(
                "id, feedback_type, title, company, note, "
                "feedback_embeddings!inner(embedding)"
            )
            .eq("user_id", user_id)
            .order("submitted_at", desc=False)
            .execute()
        )
    except Exception as e:
        logger.warning("load_feedback_embeddings failed for %s: %s", user_id, e)
        return empty

    entries = []
    for row in resp.data or []:
        embed_payload = row.get("feedback_embeddings")
        # supabase-py returns the joined record as either a dict (single match)
        # or a list with one element depending on schema; tolerate both.
        if isinstance(embed_payload, list):
            embed_payload = embed_payload[0] if embed_payload else None
        if not isinstance(embed_payload, dict):
            continue
        vec = _parse_pgvector(embed_payload.get("embedding"))
        if vec is None:
            continue
        text = format_entry_text(row)
        if not text:
            continue
        entries.append({"text": text, "embedding": vec})

    return {"entries": entries}


def ensure_feedback_embeddings(
    user_id: str,
    embed_api_key: str,
    client=None,
) -> int:
    """Embed any feedback rows that don't have an embedding yet. Return total count.

    Idempotent. Loads only the feedback IDs that already have an embedding,
    then embeds and inserts everything else. Re-running is a no-op when the
    corpus is up to date.

    Returns the user's total feedback row count so the caller can use it to
    flip the RAG_FEEDBACK_THRESHOLD switch.
    """
    if not user_id:
        return 0
    client = client or get_service_client()

    try:
        all_resp = (
            client.table("feedback")
            .select("id, feedback_type, title, company, note")
            .eq("user_id", user_id)
            .order("submitted_at", desc=False)
            .execute()
        )
        feedback_rows = all_resp.data or []
    except Exception as e:
        logger.warning("ensure_feedback_embeddings: feedback fetch failed for %s: %s", user_id, e)
        return 0

    total = len(feedback_rows)
    if total == 0:
        return 0

    try:
        emb_resp = (
            client.table("feedback_embeddings")
            .select("feedback_id")
            .eq("user_id", user_id)
            .execute()
        )
        already_embedded = {r["feedback_id"] for r in (emb_resp.data or []) if r.get("feedback_id") is not None}
    except Exception as e:
        logger.warning("ensure_feedback_embeddings: existing-embedding fetch failed for %s: %s", user_id, e)
        already_embedded = set()

    missing = [r for r in feedback_rows if r["id"] not in already_embedded]
    if not missing:
        return total

    if not embed_api_key:
        logger.warning(
            "ensure_feedback_embeddings: %d row(s) need embedding but no embed API key set "
            "for user %s — RAG will skip those entries.",
            len(missing), user_id,
        )
        return total

    # Lazy import to avoid pulling google-genai into pipelines that don't need it.
    from pipeline.core_embedding import embed_feedback_text

    new_rows = []
    skipped = 0
    for row in missing:
        text = format_entry_text(row)
        vec = embed_feedback_text(text, embed_api_key) if text else None
        if vec is None:
            skipped += 1
            continue
        new_rows.append({
            "feedback_id": row["id"],
            "user_id": user_id,
            # pgvector's text input format is "[1,2,3]"; PostgREST maps a raw
            # JSON array to a Postgres array literal "{1,2,3}", which fails the
            # vector cast. Send the bracket-string form explicitly.
            "embedding": to_pgvector_literal(vec),
        })

    if not new_rows:
        if skipped:
            logger.warning(
                "ensure_feedback_embeddings: %d row(s) couldn't be embedded (API failure) for %s.",
                skipped, user_id,
            )
        return total

    try:
        client.table("feedback_embeddings").upsert(
            new_rows, on_conflict="feedback_id"
        ).execute()
        logger.info(
            "ensure_feedback_embeddings: +%d new, %d skipped, %d total for user %s.",
            len(new_rows), skipped, total, user_id,
        )
    except Exception as e:
        logger.error(
            "ensure_feedback_embeddings: upsert FAILED for user %s (%d rows lost): %s",
            user_id, len(new_rows), e,
        )

    return total


def to_pgvector_literal(vec) -> Optional[str]:
    """Format a list of floats as pgvector's text input literal: '[0.1,0.2,...]'.

    Inverse of _parse_pgvector. Used on INSERT so PostgREST sends a string the
    vector type accepts, rather than a JSON array (which becomes a PG array
    literal and fails the cast). Returns None for an empty / non-list input.
    """
    if not isinstance(vec, list) or not vec:
        return None
    try:
        return "[" + ",".join(repr(float(x)) for x in vec) + "]"
    except (TypeError, ValueError):
        return None


def _parse_pgvector(raw) -> Optional[list]:
    """Best-effort decode of whatever shape PostgREST hands back for a vector column.

    Observed shapes in supabase-py 2.x:
      - list[float]                (rare — only with explicit casting)
      - str "[0.1,0.2,...]"        (default PostgREST JSON serialization of vector)
      - None                       (NULL embedding)

    Returns None on anything we can't safely decode so the caller drops the entry.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        if not (s.startswith("[") and s.endswith("]")):
            return None
        try:
            return [float(x) for x in s[1:-1].split(",") if x.strip()]
        except ValueError:
            return None
    return None
