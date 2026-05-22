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
PENDING_PATH = "data/feedback_pending.json"
LOG_PATH = "data/feedback_log.json"
PREFERENCES_PATH = "data/candidate_preferences.txt"

# Path in the local working tree (public code repo, edited in place).
LOCAL_REPUTATION_PATH = "data/reputation.json"

# Valid feedback types the page can submit. Anything else is dropped during
# ingestion so a hand-edited or replayed payload can't sneak garbage in.
VALID_FEEDBACK_TYPES = (
    "applied", "bookmarked", "not_relevant",
    "block_company", "wrong_location", "other",
)


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
