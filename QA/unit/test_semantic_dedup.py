"""Semantic dedup against the rolling 14-day embedding history (#5).

URL dedup catches exact repeats; semantic dedup catches the same job reposted
under a different URL by comparing embeddings. These tests verify the
threshold behaviour, the TTL pruning, and the round-trip persistence.
"""
import os
import json
from datetime import datetime, timezone, timedelta

import pandas as pd

import pipeline.core_embedding as ce
from pipeline.core_embedding import (
    cosine_similarity,
    load_embedding_history,
    save_embedding_history,
    _prune_history,
    drop_semantic_duplicates,
    update_embedding_history,
    SEMANTIC_DEDUP_THRESHOLD,
    EMBEDDING_HISTORY_TTL_DAYS,
)


def _swap_cache_path(tmp_path):
    """Helper: redirect EMBEDDING_HISTORY_CACHE to a tmp file. Returns (original, tmp)."""
    original = ce.EMBEDDING_HISTORY_CACHE
    ce.EMBEDDING_HISTORY_CACHE = tmp_path
    return original


def test_load_returns_empty_when_no_file():
    """Cold-start: no history file on disk yet -> empty dict, no error."""
    tmp = "_test_emb_missing.json"
    orig = _swap_cache_path(tmp)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        assert load_embedding_history() == {}
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig


def test_save_and_load_roundtrip():
    tmp = "_test_emb_roundtrip.json"
    orig = _swap_cache_path(tmp)
    try:
        history = {"https://co/job1": {"embedding": [0.1, 0.2, 0.3], "added_at": "2026-05-21T10:00:00+00:00"}}
        save_embedding_history(history)
        loaded = load_embedding_history()
        assert loaded["https://co/job1"]["embedding"] == [0.1, 0.2, 0.3]
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_load_recovers_from_corrupted_file():
    """A malformed JSON file should fall back to {}, not crash the pipeline."""
    tmp = "_test_emb_corrupt.json"
    orig = _swap_cache_path(tmp)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        assert load_embedding_history() == {}
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_prune_drops_old_entries():
    """Entries older than TTL get evicted; fresh ones stay."""
    now = datetime.now(tz=timezone.utc)
    old_ts = (now - timedelta(days=EMBEDDING_HISTORY_TTL_DAYS + 1)).isoformat()
    fresh_ts = (now - timedelta(days=1)).isoformat()
    history = {
        "https://old/job": {"embedding": [1.0], "added_at": old_ts},
        "https://fresh/job": {"embedding": [1.0], "added_at": fresh_ts},
    }
    _prune_history(history, now=now)
    assert "https://old/job" not in history
    assert "https://fresh/job" in history


def test_prune_drops_unparseable_timestamps():
    """Defensive: garbage in added_at -> drop the entry rather than crash later."""
    history = {"https://weird/job": {"embedding": [1.0], "added_at": "not-a-date"}}
    _prune_history(history)
    assert history == {}


def test_drop_semantic_duplicates_keeps_dissimilar_jobs():
    """A new job orthogonal to history should not be flagged as duplicate."""
    tmp = "_test_emb_dissim.json"
    orig = _swap_cache_path(tmp)
    try:
        save_embedding_history({
            "https://old/different-job": {
                "embedding": [1.0, 0.0],
                "added_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        })
        df = pd.DataFrame([{"title": "X", "job_url": "https://new/totally-different"}])
        out = drop_semantic_duplicates(df, {"https://new/totally-different": [0.0, 1.0]})
        assert len(out) == 1
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_drop_semantic_duplicates_removes_near_identical_jobs():
    """A job 0.99 cosine-similar to a history entry must be dropped."""
    tmp = "_test_emb_dup.json"
    orig = _swap_cache_path(tmp)
    try:
        save_embedding_history({
            "https://old/job": {
                "embedding": [1.0, 0.0, 0.0],
                "added_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        })
        # New embedding pointing almost the same direction: cos ~ 0.999
        df = pd.DataFrame([{"title": "Same job, different URL", "job_url": "https://new/reposted"}])
        out = drop_semantic_duplicates(df, {"https://new/reposted": [0.999, 0.01, 0.0]})
        assert len(out) == 0
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_drop_semantic_duplicates_ignores_same_url_in_history():
    """If a job URL is already in history (URL-dedup already caught it), don't
    self-match against itself — that would falsely drop every legitimate URL on
    re-runs.
    """
    tmp = "_test_emb_self.json"
    orig = _swap_cache_path(tmp)
    try:
        save_embedding_history({
            "https://same/url": {
                "embedding": [1.0, 0.0],
                "added_at": datetime.now(tz=timezone.utc).isoformat(),
            }
        })
        df = pd.DataFrame([{"title": "X", "job_url": "https://same/url"}])
        out = drop_semantic_duplicates(df, {"https://same/url": [1.0, 0.0]})
        # Row is kept — self-match is excluded from the dedup check.
        assert len(out) == 1
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_drop_semantic_duplicates_handles_empty_inputs():
    """Empty df or empty embeddings dict should short-circuit cleanly."""
    assert drop_semantic_duplicates(pd.DataFrame(), {"u1": [1.0]}).empty
    assert drop_semantic_duplicates(pd.DataFrame([{"title": "x"}]), {}).shape[0] == 1


def test_drop_semantic_duplicates_skips_when_history_empty():
    """No history yet -> nothing to compare against -> nothing dropped."""
    tmp = "_test_emb_nohistory.json"
    orig = _swap_cache_path(tmp)
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        df = pd.DataFrame([{"title": "X", "job_url": "https://a/b"}])
        out = drop_semantic_duplicates(df, {"https://a/b": [1.0, 0.0]})
        assert len(out) == 1
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig


def test_update_embedding_history_adds_timestamp():
    """New entries get an ISO timestamp so the next prune knows their age."""
    tmp = "_test_emb_update.json"
    orig = _swap_cache_path(tmp)
    try:
        update_embedding_history({"https://new/job": [0.1, 0.2]})
        loaded = load_embedding_history()
        assert "https://new/job" in loaded
        assert loaded["https://new/job"]["embedding"] == [0.1, 0.2]
        # Timestamp parses cleanly back to a datetime.
        ts = datetime.fromisoformat(loaded["https://new/job"]["added_at"])
        assert ts.tzinfo is not None
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_update_embedding_history_skips_none_values():
    """Failed embedding rows (vec=None) should not be persisted."""
    tmp = "_test_emb_skip_none.json"
    orig = _swap_cache_path(tmp)
    try:
        update_embedding_history({"https://a/ok": [0.1], "https://a/bad": None})
        loaded = load_embedding_history()
        assert "https://a/ok" in loaded
        assert "https://a/bad" not in loaded
    finally:
        ce.EMBEDDING_HISTORY_CACHE = orig
        if os.path.exists(tmp):
            os.remove(tmp)


def test_threshold_constant_is_strict():
    """Sanity check on the configured threshold — don't let it slip below 0.9."""
    assert SEMANTIC_DEDUP_THRESHOLD >= 0.9


# ---------------------------------------------------------------------------
# Injected history (multi-user path): drop_semantic_duplicates(history=...)
# The multi-user runner passes the Supabase history directly, in FLAT form
# {url: [vec]} rather than the disk form {url: {"embedding": [vec], ...}}.
# These tests lock both that the injected set is used (disk cache ignored) and
# that the flat shape is accepted.
# ---------------------------------------------------------------------------

def test_injected_flat_history_drops_repost():
    # No disk cache touched; history passed in flat form.
    df = pd.DataFrame([{"title": "Reposted role", "job_url": "https://new/url"}])
    history = {"https://old/url": [1.0, 0.0, 0.0]}
    out = drop_semantic_duplicates(df, {"https://new/url": [0.999, 0.01, 0.0]}, history=history)
    assert len(out) == 0


def test_injected_history_keeps_dissimilar():
    df = pd.DataFrame([{"title": "Different role", "job_url": "https://new/url"}])
    history = {"https://old/url": [1.0, 0.0, 0.0]}
    out = drop_semantic_duplicates(df, {"https://new/url": [0.0, 1.0, 0.0]}, history=history)
    assert len(out) == 1


def test_injected_empty_history_is_noop():
    df = pd.DataFrame([{"title": "X", "job_url": "https://a/b"}])
    out = drop_semantic_duplicates(df, {"https://a/b": [1.0, 0.0]}, history={})
    assert len(out) == 1


def test_injected_history_also_accepts_disk_shape():
    # Tolerate the {"embedding": [...]} nested shape too, for safety.
    df = pd.DataFrame([{"title": "Repost", "job_url": "https://new/url"}])
    history = {"https://old/url": {"embedding": [1.0, 0.0], "added_at": "2026-05-01T00:00:00+00:00"}}
    out = drop_semantic_duplicates(df, {"https://new/url": [1.0, 0.0]}, history=history)
    assert len(out) == 0
