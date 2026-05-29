"""B9a data-migration pure-helper tests.

The migration's correctness lives in a few pure functions: the multiset
dedup that makes it re-runnable without collapsing genuine repeat reactions,
the pgvector literal formatting, the reputation flattening, and the
defensive date parse. These are the parts a mispaired embedding or a botched
re-run would corrupt silently, so they're locked here.
"""

import migrate_to_multi_user as mig


# ---------------------------------------------------------------------------
# _feedback_key — keys both logs-repo entries and Supabase rows identically
# ---------------------------------------------------------------------------

def test_feedback_key_accepts_both_schemas():
    log_entry = {"job_url": "https://x/1", "feedback": "applied", "note": "  yes "}
    db_row = {"job_url": "https://x/1", "feedback_type": "applied", "note": "yes"}
    assert mig._feedback_key(log_entry) == mig._feedback_key(db_row)


def test_feedback_key_normalizes_missing_note_and_case():
    a = {"job_url": "https://x/1", "feedback": "Applied", "note": None}
    b = {"job_url": "https://x/1", "feedback_type": "applied"}
    assert mig._feedback_key(a) == mig._feedback_key(b)


# ---------------------------------------------------------------------------
# plan_inserts — the multiset idempotency core
# ---------------------------------------------------------------------------

def _log(url, ftype, note=""):
    return {"job_url": url, "feedback": ftype, "note": note}


def test_plan_inserts_empty_db_inserts_everything():
    log = [_log("u1", "applied"), _log("u2", "bookmarked")]
    assert mig.plan_inserts(log, []) == [0, 1]


def test_plan_inserts_skips_already_present():
    log = [_log("u1", "applied"), _log("u2", "bookmarked")]
    existing = [{"job_url": "u1", "feedback_type": "applied", "note": ""}]
    # u1/applied already there → only index 1 (u2) is new.
    assert mig.plan_inserts(log, existing) == [1]


def test_plan_inserts_is_fully_idempotent_on_rerun():
    log = [_log("u1", "applied"), _log("u2", "bookmarked"), _log("u1", "not_relevant")]
    # Simulate a completed prior run: DB now holds all three.
    existing = [
        {"job_url": "u1", "feedback_type": "applied", "note": ""},
        {"job_url": "u2", "feedback_type": "bookmarked", "note": ""},
        {"job_url": "u1", "feedback_type": "not_relevant", "note": ""},
    ]
    assert mig.plan_inserts(log, existing) == [], "re-run must insert nothing"


def test_plan_inserts_preserves_genuine_duplicate_reactions():
    # Two identical reactions in the log are real history. With an empty DB
    # both must be inserted...
    log = [_log("u1", "applied"), _log("u1", "applied")]
    assert mig.plan_inserts(log, []) == [0, 1]
    # ...and if exactly one already exists, only the second is added.
    existing = [{"job_url": "u1", "feedback_type": "applied", "note": ""}]
    assert mig.plan_inserts(log, existing) == [1]


def test_plan_inserts_resumes_a_half_finished_run():
    log = [_log("u1", "applied"), _log("u2", "bookmarked"), _log("u3", "other", "note")]
    # First run inserted the first two before dying.
    existing = [
        {"job_url": "u1", "feedback_type": "applied", "note": ""},
        {"job_url": "u2", "feedback_type": "bookmarked", "note": ""},
    ]
    assert mig.plan_inserts(log, existing) == [2]


def test_plan_inserts_note_distinguishes_other_entries():
    log = [_log("u1", "other", "too junior"), _log("u1", "other", "wrong stack")]
    existing = [{"job_url": "u1", "feedback_type": "other", "note": "too junior"}]
    # Only the differently-noted one is new.
    assert mig.plan_inserts(log, existing) == [1]


# ---------------------------------------------------------------------------
# vector_literal
# ---------------------------------------------------------------------------

def test_vector_literal_formats_floats():
    assert mig.vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"
    assert mig.vector_literal([1, 2]) == "[1.0,2.0]"


def test_vector_literal_returns_none_for_bad_input():
    assert mig.vector_literal(None) is None
    assert mig.vector_literal([]) is None
    assert mig.vector_literal("not a list") is None
    assert mig.vector_literal([0.1, "bad", 0.3]) is None


# ---------------------------------------------------------------------------
# reputation_rows
# ---------------------------------------------------------------------------

def test_reputation_rows_maps_all_three_lists():
    rep = {
        "blacklist_name_patterns": ["Skillfied", "WebBoost"],
        "blacklist_handle_patterns": ["inficore-soft"],
        "trust_boost": ["Anthropic"],
    }
    rows = mig.reputation_rows(rep, "user-1")
    by_type = {(r["pattern_type"], r["pattern"]) for r in rows}
    assert ("blacklist_name", "skillfied") in by_type
    assert ("blacklist_name", "webboost") in by_type
    assert ("blacklist_handle", "inficore-soft") in by_type
    assert ("trust_boost", "anthropic") in by_type
    assert all(r["added_by"] == "user-1" for r in rows)


def test_reputation_rows_lowercases_and_dedups():
    rep = {"trust_boost": ["Stripe", "stripe", "  STRIPE  "]}
    rows = mig.reputation_rows(rep, None)
    assert len(rows) == 1
    assert rows[0] == {"pattern_type": "trust_boost", "pattern": "stripe"}
    assert "added_by" not in rows[0], "omit added_by when no user id given"


def test_reputation_rows_handles_empty_and_garbage():
    assert mig.reputation_rows({}, "u1") == []
    assert mig.reputation_rows(None, "u1") == []
    assert mig.reputation_rows({"trust_boost": ["", "  "]}, "u1") == []


# ---------------------------------------------------------------------------
# parse_submitted_at
# ---------------------------------------------------------------------------

def test_parse_submitted_at_accepts_iso_and_z_suffix():
    assert mig.parse_submitted_at("2026-05-01T09:00:00+00:00") == "2026-05-01T09:00:00+00:00"
    # Z suffix normalized to +00:00.
    assert mig.parse_submitted_at("2026-05-01T09:00:00Z") == "2026-05-01T09:00:00+00:00"


def test_parse_submitted_at_assumes_utc_for_naive():
    out = mig.parse_submitted_at("2026-05-01T09:00:00")
    assert out.endswith("+00:00")


def test_parse_submitted_at_falls_back_to_none():
    assert mig.parse_submitted_at(None) is None
    assert mig.parse_submitted_at("") is None
    assert mig.parse_submitted_at("not a date") is None


# ---------------------------------------------------------------------------
# feedback_insert_row
# ---------------------------------------------------------------------------

def test_feedback_insert_row_shape():
    entry = {
        "job_url": "https://x/1", "title": "AI Eng", "company": "Acme",
        "feedback": "Applied", "note": "great", "date": "2026-05-01T09:00:00Z",
        "location": "Remote",  # dropped — no column; signal already in the embedding
    }
    row = mig.feedback_insert_row(entry, "user-1")
    assert row["user_id"] == "user-1"
    assert row["job_result_id"] is None
    assert row["job_url"] == "https://x/1"
    assert row["feedback_type"] == "applied"
    assert row["title"] == "AI Eng"
    assert row["note"] == "great"
    assert row["submitted_at"] == "2026-05-01T09:00:00+00:00"
    assert "location" not in row


def test_feedback_insert_row_omits_submitted_at_when_unparseable():
    entry = {"job_url": "https://x/1", "feedback": "other", "date": "garbage"}
    row = mig.feedback_insert_row(entry, "user-1")
    assert "submitted_at" not in row
    assert row["title"] is None and row["company"] is None and row["note"] is None
