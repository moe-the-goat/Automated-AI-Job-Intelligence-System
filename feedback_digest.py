import json
import os

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_feedback import (
    LOG_PATH,
    PREFERENCES_PATH,
    RAG_FEEDBACK_THRESHOLD,
    LogsRepoAuthError,
    _read_file,
    _write_file,
    format_entry_text,
    verify_logs_repo_access,
)

logger = get_logger(__name__)

"""
FEEDBACK DIGEST
---------------
Runs on a separate cron (every 5 days). Reads `data/feedback_log.json` from
the private logs repo, asks the strongest available LLM (Cerebras with Groq
fallback) to compress the entries into a 5-8 sentence candidate preference
profile, and writes the summary back to `data/candidate_preferences.txt`.
The next pipeline run picks up the refreshed profile and injects it into
every verdict prompt.

Once the log crosses RAG_FEEDBACK_THRESHOLD entries the digest stops running
— the scraper switches to per-job retrieval against `feedback_embeddings.json`
and the global summary becomes dead weight. The early exit below makes that
switch automatic so nothing has to be removed by hand on the cutover day.

The raw log is NEVER trimmed — every reaction the user has ever submitted
stays in `feedback_log.json` indefinitely, both to keep RAG's corpus intact
and so we can re-derive the digest if we ever need to roll back.
"""


SUMMARY_PROMPT = """You are analyzing a candidate's job-search feedback history to build a
deep, useful preference profile. The profile will be injected into a recruiter-screen
prompt that scores future job listings against the candidate's CV. Your output IS
the profile — write it as if a thoughtful recruiter is describing the candidate's
preferences to another recruiter.

Each feedback entry below records the candidate's reaction to one job that the
pipeline surfaced. Possible feedback types:

  applied         — submitted an application; treat as a clear positive signal
  bookmarked      — interested but did not apply yet
  not_relevant    — irrelevant to the search; treat as a clear negative signal
  block_company   — never show this company again
  wrong_location  — geographically incompatible (look for location patterns)
  other           — read the note; the candidate left a free-form explanation

FEEDBACK HISTORY (chronological, most recent last):
{entries}

Write a CANDIDATE PREFERENCE PROFILE in 5 to 8 sentences. Cover:
  1. Role categories the candidate consistently engages with.
  2. Role categories they consistently reject (and why, if the notes say).
  3. Specific companies, industries, or company-size patterns they block or favor.
  4. Tech-stack or seniority patterns visible in the notes and feedback.
  5. Inferred preferences future scoring should weight more or less.

Be concrete and specific. Cite role types, technologies, and company traits by name.
Do not invent information not present in the entries. Output prose only — no JSON,
no bullet lists, no markdown headers.
"""


def run_digest():
    repo = os.environ.get("LOGS_REPO")
    token = os.environ.get("LOGS_REPO_TOKEN")
    cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")

    if not repo or not token:
        logger.error("LOGS_REPO and LOGS_REPO_TOKEN are required to run the digest.")
        return False
    if not cerebras_key and not groq_key:
        logger.error("CEREBRAS_API_KEY or GROQ_API_KEY required for summarization.")
        return False

    if not verify_logs_repo_access(repo, token):
        logger.error("Digest: logs repo unreachable — aborting (see CRITICAL log above for remediation).")
        return False

    try:
        log_text, _ = _read_file(repo, LOG_PATH, token)
    except LogsRepoAuthError as e:
        logger.critical("Digest: %s", e)
        return False
    if not log_text:
        logger.info("Digest: feedback log missing or empty, nothing to summarize.")
        return True

    try:
        log_data = json.loads(log_text)
    except json.JSONDecodeError as e:
        logger.error("Digest: feedback log malformed (%s), aborting.", e)
        return False

    entries = log_data.get("entries", []) if isinstance(log_data, dict) else []
    if not entries:
        logger.info("Digest: feedback log has no entries.")
        return True

    # Once the log crosses the RAG threshold the scraper stops reading
    # candidate_preferences.txt — running the digest LLM call would burn
    # tokens to produce a file no one consumes. Bail out cleanly so the
    # cron job becomes a no-op without anyone having to disable it manually.
    if len(entries) >= RAG_FEEDBACK_THRESHOLD:
        logger.info(
            "Digest: %d entries >= RAG threshold %d. Scraper is on per-job retrieval now — skipping summary.",
            len(entries), RAG_FEEDBACK_THRESHOLD,
        )
        return True

    formatted = "\n".join(format_entry_text(e) for e in entries)
    prompt = SUMMARY_PROMPT.format(entries=formatted)

    from pipeline.core_llm import call_llm_with_fallback  # lazy

    try:
        summary = call_llm_with_fallback(
            prompt,
            cerebras_key=cerebras_key,
            groq_key=groq_key,
            max_attempts=4,
            label="feedback-digest",
        )
    except Exception as e:
        logger.error("Digest: LLM call failed (%s), aborting.", e)
        return False

    summary = (summary or "").strip()
    if not summary:
        logger.error("Digest: LLM returned an empty summary, aborting.")
        return False

    try:
        _, pref_sha = _read_file(repo, PREFERENCES_PATH, token)
        _write_file(
            repo, PREFERENCES_PATH, summary, pref_sha, token,
            "Refreshed candidate preference profile from feedback log",
        )
    except LogsRepoAuthError as e:
        logger.critical("Digest: failed to persist preferences: %s", e)
        return False

    # `feedback_log.json` is left untouched — see module docstring on why
    # the log is never trimmed.

    logger.info("Digest: summarized %d entries; preference profile refreshed.", len(entries))
    return True


def main():
    configure_logging()
    ok = run_digest()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
