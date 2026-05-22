import json
import os

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_feedback import (
    LOG_PATH,
    PREFERENCES_PATH,
    _read_file,
    _write_file,
)

logger = get_logger(__name__)

"""
FEEDBACK DIGEST
---------------
Runs on a separate cron (every two weeks). Reads `data/feedback_log.json`
from the private logs repo, asks the strongest available LLM (Cerebras with
Groq fallback) to compress the entries into a 5-8 sentence candidate
preference profile, writes the summary back to
`data/candidate_preferences.txt`, and trims the raw log so it doesn't grow
without bound. The next pipeline run picks up the refreshed profile and
injects it into every verdict prompt.
"""

# Keep a small tail of recent entries after a digest so the next summary
# still has fresh ground truth even if no feedback arrives between digests.
LOG_RETAINED_TAIL = 3


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


def _format_entry(e):
    parts = [f"[{e.get('feedback', 'unknown')}]", f"{e.get('title', '?')} @ {e.get('company', '?')}"]
    if e.get('location'):
        parts.append(f"({e['location']})")
    if e.get('note'):
        parts.append(f"— note: {e['note']}")
    return " ".join(parts)


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

    log_text, log_sha = _read_file(repo, LOG_PATH, token)
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

    formatted = "\n".join(_format_entry(e) for e in entries)
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

    _, pref_sha = _read_file(repo, PREFERENCES_PATH, token)
    _write_file(
        repo, PREFERENCES_PATH, summary, pref_sha, token,
        "Refreshed candidate preference profile from feedback log",
    )

    trimmed = {"entries": entries[-LOG_RETAINED_TAIL:]}
    _write_file(
        repo, LOG_PATH, json.dumps(trimmed, indent=2), log_sha, token,
        f"Trimmed feedback log after digest (kept last {LOG_RETAINED_TAIL} entries)",
    )

    logger.info("Digest: summarized %d entries; preference profile refreshed.", len(entries))
    return True


def main():
    configure_logging()
    ok = run_digest()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
