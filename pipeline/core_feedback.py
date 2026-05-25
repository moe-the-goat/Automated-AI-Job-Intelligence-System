import base64
import json
import os
import re

import requests

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)

"""
CORE FEEDBACK MODULE
--------------------
Reads the user's per-job feedback signals from the previous day, applies any
hard rules they imply (auto-blacklisting blocked companies), appends every
entry to a rolling history log, and exposes the latest AI-summarized
preference profile so the verdict prompt can read it.

All personal feedback data lives in the PRIVATE logs repo and is accessed
through the GitHub Contents API. The PUBLIC code repo only changes when a
feedback entry mutates a deterministic filter file (currently just
`data/reputation.json` on a `block_company` signal).
"""

# Paths inside the private logs repo, addressed via the Contents API.
# `feedback_pending.json` is the inbox: the Cloudflare Worker writes to it
# when the user submits the feedback page; the pipeline drains it the next
# morning. The log and preferences files are written by the pipeline.
PENDING_PATH = "data/feedback_pending.json"
LOG_PATH = "data/feedback_log.json"
PREFERENCES_PATH = "data/candidate_preferences.txt"
EMBEDDINGS_PATH = "data/feedback_embeddings.json"

# Path in the local working tree (public code repo, edited in place).
LOCAL_REPUTATION_PATH = "data/reputation.json"

# Valid feedback types the page can submit. Anything else is dropped during
# ingestion so a hand-edited or replayed payload can't sneak garbage in.
VALID_FEEDBACK_TYPES = (
    "applied", "bookmarked", "not_relevant",
    "block_company", "wrong_location", "other",
)

# RAG switch (2026-05-25). Below this many archived feedback entries the
# pipeline uses the 5-day digest summary as the candidate-preference context
# for every job (cheap, deterministic, identical signal for every verdict).
# At or above this many entries the pipeline switches to per-job retrieval:
# embed the job, find the top-K most-similar past feedback entries, inject
# those as preferences. The digest run becomes dead-but-harmless on the same
# tick — `feedback_digest.py` early-exits and `candidate_preferences.txt` is
# no longer read.
#
# RAG_TOP_K controls how many past entries to inject. 5 keeps the prompt
# focused; raising it dilutes the signal with marginal matches.
RAG_FEEDBACK_THRESHOLD = 50
RAG_TOP_K = 5


def _normalize_repo(repo):
    """Coerce `owner/name`, full URL, or `github.com/owner/name` into `owner/name`."""
    if not repo:
        return None
    s = str(repo).strip().rstrip("/")
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^(www\.)?github\.com/", "", s)
    if s.endswith(".git"):
        s = s[:-4]
    return s or None


def _api_url(repo, path):
    return f"https://api.github.com/repos/{repo}/contents/{path}"


def _read_file(repo, path, token):
    """Read a single file from the private logs repo via the Contents API.

    Returns (text, sha) on success, (None, None) when the file is missing or
    on any transport error. Callers treat (None, None) as "no data yet".
    """
    repo = _normalize_repo(repo)
    if not repo or not token:
        return None, None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        r = requests.get(_api_url(repo, path), headers=headers, timeout=15)
        if r.status_code == 404:
            return None, None
        r.raise_for_status()
        data = r.json()
        text = base64.b64decode(data["content"]).decode("utf-8")
        return text, data.get("sha")
    except Exception as e:
        logger.warning("Feedback: failed to read %s from %s: %s", path, repo, e)
        return None, None


def _write_file(repo, path, content, sha, token, message):
    """Create or update a single file via the Contents API. Returns True on success."""
    repo = _normalize_repo(repo)
    if not repo or not token:
        return False
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    try:
        r = requests.put(_api_url(repo, path), headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Feedback: failed to write %s to %s: %s", path, repo, e)
        return False


def load_candidate_preferences(repo, token):
    """Return the latest AI-summarized preference profile, or empty string.

    Loaded once per pipeline run and injected into every verdict prompt.
    """
    text, _ = _read_file(repo, PREFERENCES_PATH, token)
    return (text or "").strip()


def _sanitize_entries(raw_entries):
    """Drop malformed entries and clamp string fields to safe lengths.

    The feedback page submits a JSON shape we control, but the page sits on a
    public URL with an embedded write token. A bad actor (or a corrupted
    payload) shouldn't be able to break ingestion just by hand-editing the
    file. We accept only entries with a known `feedback` type and a non-empty
    `job_url`, then trim every text field to a reasonable cap.
    """
    if not isinstance(raw_entries, list):
        return []
    out = []
    for e in raw_entries:
        if not isinstance(e, dict):
            continue
        fb = str(e.get("feedback", "")).strip().lower()
        if fb not in VALID_FEEDBACK_TYPES:
            continue
        url = str(e.get("job_url", "")).strip()
        if not url:
            continue
        out.append({
            "job_url": url[:500],
            "company": str(e.get("company", ""))[:200],
            "title": str(e.get("title", ""))[:200],
            "location": str(e.get("location", ""))[:200],
            "feedback": fb,
            "note": str(e.get("note", ""))[:1000],
            "date": str(e.get("date", ""))[:32],
        })
    return out


def _apply_block_companies(entries):
    """Append any `block_company` company names to `data/reputation.json`.

    Returns the count of new entries added. Existing entries are skipped, so
    repeated feedback on the same company is idempotent.
    """
    blocks = [
        e["company"].strip().lower()
        for e in entries
        if e["feedback"] == "block_company" and e.get("company", "").strip()
    ]
    if not blocks:
        return 0

    try:
        with open(LOCAL_REPUTATION_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as ex:
        logger.warning("reputation.json missing or malformed (%s), skipping block_company updates.", ex)
        return 0

    existing = {p.lower() for p in rep.get("blacklist_name_patterns", [])}
    new = []
    seen = set()
    for c in blocks:
        if c in existing or c in seen:
            continue
        seen.add(c)
        new.append(c)
    if not new:
        return 0

    rep.setdefault("blacklist_name_patterns", []).extend(new)
    try:
        with open(LOCAL_REPUTATION_PATH, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        logger.info("Feedback: added %d company name(s) to reputation blacklist: %s",
                    len(new), ", ".join(new))
        return len(new)
    except OSError as ex:
        logger.warning("Failed to write reputation.json: %s", ex)
        return 0


def _mark_applied_seen(entries, tracker):
    """Mark `applied` feedback URLs as seen in the JobTracker.

    Defensive — the URL was almost certainly already marked during AI eval,
    but if the user got the email a day late and the cache rolled over, this
    keeps the job from reappearing.
    """
    if tracker is None:
        return
    for e in entries:
        if e["feedback"] == "applied":
            tracker.mark_seen(e["job_url"])


def ingest_pending_feedback(repo, token, tracker=None):
    """Drain `feedback_pending.json` from the logs repo, apply signals, archive entries.

    Steps:
      1. Read pending file. Missing or empty → no-op.
      2. Sanitize entries (drop malformed, clamp field lengths).
      3. Apply hard signals (block_company → reputation.json, applied → tracker).
      4. Append sanitized entries to `feedback_log.json`.
      5. Clear `feedback_pending.json` so tomorrow starts empty.

    Returns the number of valid entries ingested. Pure no-op when LOGS_REPO /
    LOGS_REPO_TOKEN aren't configured.
    """
    if not repo or not token:
        logger.info("Feedback: LOGS_REPO/LOGS_REPO_TOKEN missing, skipping ingestion.")
        return 0

    pending_text, pending_sha = _read_file(repo, PENDING_PATH, token)
    if not pending_text:
        logger.info("Feedback: no pending entries.")
        return 0

    try:
        payload = json.loads(pending_text)
    except json.JSONDecodeError as e:
        logger.warning("Feedback: pending file malformed JSON (%s), skipping ingestion.", e)
        return 0

    raw_entries = payload.get("entries", []) if isinstance(payload, dict) else []
    entries = _sanitize_entries(raw_entries)
    if not entries:
        logger.info("Feedback: pending file had no valid entries, clearing it.")
        _write_file(repo, PENDING_PATH, json.dumps({"entries": []}, indent=2),
                    pending_sha, token, "Cleared empty feedback pending")
        return 0

    rep_updates = _apply_block_companies(entries)
    _mark_applied_seen(entries, tracker)

    log_text, log_sha = _read_file(repo, LOG_PATH, token)
    log_data = {"entries": []}
    if log_text:
        try:
            parsed = json.loads(log_text)
            if isinstance(parsed, dict) and isinstance(parsed.get("entries"), list):
                log_data = parsed
        except json.JSONDecodeError:
            logger.warning("Feedback log malformed, restarting from empty.")

    log_data["entries"].extend(entries)
    _write_file(
        repo, LOG_PATH, json.dumps(log_data, indent=2),
        log_sha, token,
        f"Appended {len(entries)} feedback entries to log",
    )
    _write_file(
        repo, PENDING_PATH, json.dumps({"entries": []}, indent=2),
        pending_sha, token, "Cleared ingested feedback pending",
    )

    logger.info(
        "Feedback: ingested %d entr%s (%d new blacklist entries, %d total in log).",
        len(entries), "y" if len(entries) == 1 else "ies",
        rep_updates, len(log_data["entries"]),
    )
    return len(entries)


# ---------------------------------------------------------------------------
# Feedback log → embeddings → RAG switch (2026-05-25)
# ---------------------------------------------------------------------------

def format_entry_text(entry):
    """Canonical short-text representation of a feedback entry.

    Used both as input to the embedding model and as the human-readable text
    the verdict prompt sees at retrieval time. Kept compact on purpose — the
    embedding model maps short, consistent phrases to dense vectors much more
    cleanly than long unstructured notes.
    """
    if not isinstance(entry, dict):
        return ""
    parts = [
        f"[{entry.get('feedback', 'unknown')}]",
        f"{entry.get('title', '?')} @ {entry.get('company', '?')}",
    ]
    if entry.get("location"):
        parts.append(f"({entry['location']})")
    if entry.get("note"):
        parts.append(f"— note: {entry['note']}")
    return " ".join(parts)


def _load_log_entries(repo, token):
    """Read feedback_log.json and return its entries list (or [] on miss/error)."""
    log_text, _ = _read_file(repo, LOG_PATH, token)
    if not log_text:
        return []
    try:
        data = json.loads(log_text)
    except json.JSONDecodeError:
        logger.warning("Feedback log malformed — treating as empty.")
        return []
    if not isinstance(data, dict):
        return []
    entries = data.get("entries", [])
    return entries if isinstance(entries, list) else []


def count_feedback_entries(repo, token):
    """Return the number of entries currently in feedback_log.json.

    Drives the RAG switch in scraper.py / local_companies.py — once this count
    crosses RAG_FEEDBACK_THRESHOLD the pipeline routes to per-job retrieval
    instead of the global digest summary.
    """
    return len(_load_log_entries(repo, token))


def load_feedback_embeddings(repo, token):
    """Return the parsed feedback_embeddings.json contents, or {"entries": []}.

    Shape: {"entries": [{"text": str, "embedding": list[float]}, ...]}.
    Entry order matches feedback_log.json's entries array — `ensure_feedback_embeddings`
    is what keeps the two in sync.
    """
    text, _ = _read_file(repo, EMBEDDINGS_PATH, token)
    if not text:
        return {"entries": []}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("feedback_embeddings.json malformed — treating as empty.")
        return {"entries": []}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {"entries": []}
    return data


def ensure_feedback_embeddings(repo, token, embed_api_key):
    """Embed any feedback-log entries that don't yet have an embedding, persist, return total count.

    Idempotent: comparing lengths of log.entries vs embeddings.entries tells us
    exactly which suffix is missing, so re-running is a no-op when up-to-date.
    The two files stay positionally aligned — index i in embeddings.entries
    corresponds to index i in log.entries.

    Returns the entry count (used to decide RAG vs digest mode). On any error
    the function logs and returns the count it WAS able to determine — never
    raises into the pipeline.
    """
    if not repo or not token:
        return 0

    log_entries = _load_log_entries(repo, token)
    total = len(log_entries)
    if total == 0:
        return 0

    embeddings = load_feedback_embeddings(repo, token)
    embedded_entries = embeddings.get("entries", [])
    existing = len(embedded_entries)

    if existing >= total:
        # Already up to date (or there was a manual trim — we don't try to fix that).
        return total

    if not embed_api_key:
        logger.warning(
            "Feedback embeddings: %d entries unembedded but no embed API key — RAG retrieval will skip them.",
            total - existing,
        )
        return total

    # Lazy import — only loaded when there's actual embedding work to do.
    from pipeline.core_embedding import embed_feedback_text

    new_count = 0
    skipped = 0
    for entry in log_entries[existing:]:
        text = format_entry_text(entry)
        vec = embed_feedback_text(text, embed_api_key) if text else None
        if vec is None:
            # Persist a placeholder so the position alignment with log_entries
            # is preserved; retrieval skips entries with no vector.
            embedded_entries.append({"text": text, "embedding": None})
            skipped += 1
            continue
        embedded_entries.append({"text": text, "embedding": vec})
        new_count += 1

    if new_count == 0 and skipped == 0:
        return total

    embeddings["entries"] = embedded_entries
    _, emb_sha = _read_file(repo, EMBEDDINGS_PATH, token)
    ok = _write_file(
        repo, EMBEDDINGS_PATH,
        json.dumps(embeddings, indent=2),
        emb_sha, token,
        f"Embedded {new_count} new feedback entr{'y' if new_count == 1 else 'ies'}",
    )
    if ok:
        logger.info(
            "Feedback embeddings: +%d new (%d skipped on API failure), %d total now indexed.",
            new_count, skipped, len(embedded_entries),
        )
    return total
