"""Multi-user digest threshold gate (B7c).

The digest's whole job is to STOP cleanly once a user crosses the RAG
threshold — past that point the runner uses per-job retrieval and a
regenerated global profile is dead weight (wasted LLM tokens). That gate
reads profiles.feedback_count via count_feedback_entries(), which is what
migration 0006's trigger keeps honest.

These tests lock run_for_user's branching so a future refactor can't
silently resurrect the "digest runs forever past threshold" bug:
  * 0 entries          -> skip, timestamp-only bump, no LLM call
  * >= threshold        -> skip, timestamp-only bump, no LLM call
  * below threshold     -> summarize + persist
  * empty summary       -> return False (surfaces in CI), no persist
  * counter/table drift -> tolerated (no crash)
"""

import feedback_digest_multi_user as fdm
from pipeline.core_feedback_supabase import RAG_FEEDBACK_THRESHOLD


class _Recorder:
    """Captures which side-effects run_for_user triggered."""

    def __init__(self):
        self.summarized = False
        self.persisted = None
        self.timestamp_bumped = False
        self.fetched = False


def _install(monkey, recorder, *, count, entries, summary):
    """Monkeypatch the module's collaborators. Returns a restore callable."""
    originals = {
        "count_feedback_entries": fdm.count_feedback_entries,
        "_fetch_user_feedback": fdm._fetch_user_feedback,
        "_summarize": fdm._summarize,
        "_persist_summary": fdm._persist_summary,
        "_bump_digest_timestamp_only": fdm._bump_digest_timestamp_only,
    }

    def fake_count(user_id, client=None):
        return count

    def fake_fetch(client, user_id):
        recorder.fetched = True
        return entries

    def fake_summarize(entries_arg, *, cerebras_key, groq_key):
        recorder.summarized = True
        return summary

    def fake_persist(client, user_id, text):
        recorder.persisted = text
        return True

    def fake_bump(client, user_id):
        recorder.timestamp_bumped = True

    fdm.count_feedback_entries = fake_count
    fdm._fetch_user_feedback = fake_fetch
    fdm._summarize = fake_summarize
    fdm._persist_summary = fake_persist
    fdm._bump_digest_timestamp_only = fake_bump

    def restore():
        for name, fn in originals.items():
            setattr(fdm, name, fn)

    return restore


def _run(recorder, *, count, entries, summary):
    restore = _install(None, recorder, count=count, entries=entries, summary=summary)
    try:
        return fdm.run_for_user(client=object(), user_id="u1",
                                cerebras_key="ck", groq_key="gk")
    finally:
        restore()


def test_zero_entries_skips_without_llm():
    rec = _Recorder()
    ok = _run(rec, count=0, entries=[], summary="should not be used")
    assert ok is True
    assert rec.summarized is False, "must not call the LLM for a user with no feedback"
    assert rec.persisted is None
    assert rec.timestamp_bumped is True


def test_at_or_above_threshold_skips_without_llm():
    rec = _Recorder()
    ok = _run(rec, count=RAG_FEEDBACK_THRESHOLD, entries=[{"x": 1}], summary="unused")
    assert ok is True
    assert rec.summarized is False, "past the RAG threshold the digest must NOT regenerate a profile"
    assert rec.persisted is None
    assert rec.timestamp_bumped is True


def test_just_below_threshold_summarizes_and_persists():
    rec = _Recorder()
    entries = [{"feedback_type": "applied", "title": "Eng", "company": "Acme"}]
    ok = _run(rec, count=RAG_FEEDBACK_THRESHOLD - 1, entries=entries, summary="A real profile.")
    assert ok is True
    assert rec.summarized is True
    assert rec.persisted == "A real profile."
    assert rec.timestamp_bumped is False, "below threshold we persist a profile, not just bump the clock"


def test_empty_summary_returns_false_and_does_not_persist():
    rec = _Recorder()
    entries = [{"feedback_type": "applied", "title": "Eng", "company": "Acme"}]
    ok = _run(rec, count=10, entries=entries, summary=None)
    assert ok is False, "an empty/failed summary must surface as a failure in CI"
    assert rec.persisted is None


def test_counter_table_drift_is_tolerated():
    # profiles.feedback_count says there's feedback, but the table returns none
    # (e.g. a half-applied delete). Must not crash or persist garbage.
    rec = _Recorder()
    ok = _run(rec, count=12, entries=[], summary="unused")
    assert ok is True
    assert rec.summarized is False
    assert rec.persisted is None
