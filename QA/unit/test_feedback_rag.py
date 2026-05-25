"""Feedback RAG tests: per-job retrieval, entry counting, idempotent embedding sync.

Network calls (Gemini Embedding 2 + GitHub Contents API) are stubbed so the
suite stays offline. The pure-math part of retrieval reuses cosine_similarity
from core_embedding, which already has its own coverage in test_embedding_math.
"""

import json

import pipeline.core_feedback as cf
import pipeline.core_embedding as ce
from pipeline.core_feedback import (
    RAG_FEEDBACK_THRESHOLD,
    RAG_TOP_K,
    format_entry_text,
    count_feedback_entries,
    load_feedback_embeddings,
    ensure_feedback_embeddings,
)
from pipeline.core_embedding import retrieve_relevant_feedback


# ---------------------------------------------------------------------------
# format_entry_text — canonical text for both embedding and digest input
# ---------------------------------------------------------------------------

def test_format_entry_text_minimal():
    text = format_entry_text({"feedback": "applied", "title": "Backend Engineer", "company": "Stripe"})
    assert "[applied]" in text
    assert "Backend Engineer @ Stripe" in text


def test_format_entry_text_with_location_and_note():
    text = format_entry_text({
        "feedback": "bookmarked",
        "title": "ML Engineer",
        "company": "Anthropic",
        "location": "Remote",
        "note": "looks great",
    })
    assert "[bookmarked]" in text
    assert "ML Engineer @ Anthropic" in text
    assert "(Remote)" in text
    assert "looks great" in text


def test_format_entry_text_missing_title_company():
    text = format_entry_text({"feedback": "other"})
    assert "? @ ?" in text


def test_format_entry_text_non_dict_returns_empty():
    assert format_entry_text(None) == ""
    assert format_entry_text("garbage") == ""
    assert format_entry_text(42) == ""


# ---------------------------------------------------------------------------
# Stubbed GitHub API helpers — small fake store keyed by (repo, path)
# ---------------------------------------------------------------------------

class _FakeRepo:
    """Replaces _read_file / _write_file with an in-memory dict.

    Stdlib-only stand-in for the GitHub Contents API. Tracks SHA so write paths
    can pretend they're updating a known revision.
    """

    def __init__(self, seed=None):
        self.store = dict(seed or {})
        self.writes = []

    def read(self, repo, path, token):
        key = (repo, path)
        if key not in self.store:
            return None, None
        return self.store[key], f"sha-{len(self.store[key])}"

    def write(self, repo, path, content, sha, token, message):
        self.store[(repo, path)] = content
        self.writes.append((repo, path, message, len(content)))
        return True


def _patch_repo(monkey=None):
    """Install a _FakeRepo and patch core_feedback's _read_file/_write_file.

    Returns (fake, restore). Caller must call restore() in a finally.
    """
    fake = _FakeRepo()
    orig_read = cf._read_file
    orig_write = cf._write_file
    cf._read_file = fake.read
    cf._write_file = fake.write
    def restore():
        cf._read_file = orig_read
        cf._write_file = orig_write
    return fake, restore


# ---------------------------------------------------------------------------
# count_feedback_entries
# ---------------------------------------------------------------------------

def test_count_feedback_entries_zero_when_missing():
    fake, restore = _patch_repo()
    try:
        assert count_feedback_entries("owner/repo", "token") == 0
    finally:
        restore()


def test_count_feedback_entries_zero_when_malformed():
    fake, restore = _patch_repo()
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = "{ not json"
        assert count_feedback_entries("owner/repo", "token") == 0
    finally:
        restore()


def test_count_feedback_entries_reads_entries_list():
    fake, restore = _patch_repo()
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [{"feedback": "applied", "job_url": "u1"} for _ in range(7)]
        })
        assert count_feedback_entries("owner/repo", "token") == 7
    finally:
        restore()


def test_count_feedback_entries_handles_missing_token():
    fake, restore = _patch_repo()
    try:
        assert count_feedback_entries("", "") == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# load_feedback_embeddings
# ---------------------------------------------------------------------------

def test_load_feedback_embeddings_empty_when_missing():
    fake, restore = _patch_repo()
    try:
        assert load_feedback_embeddings("owner/repo", "token") == {"entries": []}
    finally:
        restore()


def test_load_feedback_embeddings_returns_parsed_dict():
    fake, restore = _patch_repo()
    try:
        payload = {"entries": [{"text": "[applied] X @ Y", "embedding": [0.1, 0.2]}]}
        fake.store[("owner/repo", cf.EMBEDDINGS_PATH)] = json.dumps(payload)
        out = load_feedback_embeddings("owner/repo", "token")
        assert out["entries"][0]["text"] == "[applied] X @ Y"
        assert out["entries"][0]["embedding"] == [0.1, 0.2]
    finally:
        restore()


def test_load_feedback_embeddings_recovers_from_malformed_json():
    fake, restore = _patch_repo()
    try:
        fake.store[("owner/repo", cf.EMBEDDINGS_PATH)] = "{ not json"
        assert load_feedback_embeddings("owner/repo", "token") == {"entries": []}
    finally:
        restore()


# ---------------------------------------------------------------------------
# ensure_feedback_embeddings — idempotent backfill
# ---------------------------------------------------------------------------

def test_ensure_feedback_embeddings_no_op_when_log_empty():
    fake, restore = _patch_repo()
    try:
        total = ensure_feedback_embeddings("owner/repo", "token", "fake-embed-key")
        assert total == 0
        assert fake.writes == []
    finally:
        restore()


def test_ensure_feedback_embeddings_embeds_all_when_no_embeddings_file():
    fake, restore = _patch_repo()
    orig_embed = ce.embed_feedback_text
    try:
        # Seed 3 log entries, no embeddings file yet.
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [
                {"feedback": "applied", "title": "A", "company": "Co1"},
                {"feedback": "bookmarked", "title": "B", "company": "Co2"},
                {"feedback": "not_relevant", "title": "C", "company": "Co3"},
            ]
        })
        # Stub the network embed call.
        ce.embed_feedback_text = lambda text, key: [0.1, 0.2, 0.3]
        total = ensure_feedback_embeddings("owner/repo", "token", "fake-embed-key")
        assert total == 3

        stored = json.loads(fake.store[("owner/repo", cf.EMBEDDINGS_PATH)])
        assert len(stored["entries"]) == 3
        assert all(e["embedding"] == [0.1, 0.2, 0.3] for e in stored["entries"])
        assert stored["entries"][0]["text"].startswith("[applied]")
    finally:
        ce.embed_feedback_text = orig_embed
        restore()


def test_ensure_feedback_embeddings_only_embeds_missing_suffix():
    """If 5 entries are already embedded and 2 new arrived, embed only those 2."""
    fake, restore = _patch_repo()
    orig_embed = ce.embed_feedback_text
    embed_calls = []
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [{"feedback": "applied", "title": f"T{i}", "company": "Co"} for i in range(7)]
        })
        existing = {"entries": [{"text": f"[applied] T{i} @ Co", "embedding": [0.0, float(i)]} for i in range(5)]}
        fake.store[("owner/repo", cf.EMBEDDINGS_PATH)] = json.dumps(existing)

        def _stub_embed(text, key):
            embed_calls.append(text)
            return [1.0, 1.0]
        ce.embed_feedback_text = _stub_embed

        total = ensure_feedback_embeddings("owner/repo", "token", "fake-embed-key")
        assert total == 7
        # Exactly 2 new embed calls — for entries 5 and 6.
        assert len(embed_calls) == 2
        assert "T5" in embed_calls[0] and "T6" in embed_calls[1]

        stored = json.loads(fake.store[("owner/repo", cf.EMBEDDINGS_PATH)])
        assert len(stored["entries"]) == 7
        # First 5 untouched.
        assert stored["entries"][0]["embedding"] == [0.0, 0.0]
        # Newly embedded.
        assert stored["entries"][5]["embedding"] == [1.0, 1.0]
        assert stored["entries"][6]["embedding"] == [1.0, 1.0]
    finally:
        ce.embed_feedback_text = orig_embed
        restore()


def test_ensure_feedback_embeddings_is_noop_when_already_up_to_date():
    fake, restore = _patch_repo()
    orig_embed = ce.embed_feedback_text
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [{"feedback": "applied", "title": "T", "company": "C"}]
        })
        fake.store[("owner/repo", cf.EMBEDDINGS_PATH)] = json.dumps({
            "entries": [{"text": "[applied] T @ C", "embedding": [0.5, 0.5]}]
        })
        ce.embed_feedback_text = lambda text, key: (_ for _ in ()).throw(AssertionError("should not embed"))
        total = ensure_feedback_embeddings("owner/repo", "token", "fake-embed-key")
        assert total == 1
        # No new writes — already up to date.
        assert fake.writes == []
    finally:
        ce.embed_feedback_text = orig_embed
        restore()


def test_ensure_feedback_embeddings_stores_none_on_api_failure_to_preserve_alignment():
    """If the embed API returns None for one entry, the slot still gets stored so
    later entries keep their correct index — retrieval just skips None vectors."""
    fake, restore = _patch_repo()
    orig_embed = ce.embed_feedback_text
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [
                {"feedback": "applied", "title": "OK1", "company": "C"},
                {"feedback": "applied", "title": "FAIL", "company": "C"},
                {"feedback": "applied", "title": "OK2", "company": "C"},
            ]
        })
        call_count = {"n": 0}
        def _stub(text, key):
            call_count["n"] += 1
            return None if "FAIL" in text else [0.7, 0.7]
        ce.embed_feedback_text = _stub
        total = ensure_feedback_embeddings("owner/repo", "token", "fake-embed-key")
        assert total == 3
        stored = json.loads(fake.store[("owner/repo", cf.EMBEDDINGS_PATH)])
        assert len(stored["entries"]) == 3
        assert stored["entries"][0]["embedding"] == [0.7, 0.7]
        assert stored["entries"][1]["embedding"] is None
        assert stored["entries"][2]["embedding"] == [0.7, 0.7]
    finally:
        ce.embed_feedback_text = orig_embed
        restore()


def test_ensure_feedback_embeddings_skips_when_no_api_key():
    """Without an embed key the function still reports the entry count so the RAG
    switch decision is correct, but doesn't attempt to embed."""
    fake, restore = _patch_repo()
    orig_embed = ce.embed_feedback_text
    try:
        fake.store[("owner/repo", cf.LOG_PATH)] = json.dumps({
            "entries": [{"feedback": "applied", "title": "T", "company": "C"} for _ in range(60)]
        })
        ce.embed_feedback_text = lambda text, key: (_ for _ in ()).throw(AssertionError("no embed"))
        total = ensure_feedback_embeddings("owner/repo", "token", "")
        assert total == 60
        assert ("owner/repo", cf.EMBEDDINGS_PATH) not in fake.store
    finally:
        ce.embed_feedback_text = orig_embed
        restore()


def test_ensure_feedback_embeddings_returns_zero_without_repo_or_token():
    fake, restore = _patch_repo()
    try:
        assert ensure_feedback_embeddings("", "token", "key") == 0
        assert ensure_feedback_embeddings("owner/repo", "", "key") == 0
    finally:
        restore()


# ---------------------------------------------------------------------------
# retrieve_relevant_feedback — top-K cosine-ranked formatting
# ---------------------------------------------------------------------------

def _job(title="ML Engineer", company="X", description="train neural networks"):
    return {"title": title, "company": company, "description": description, "location": ""}


def test_retrieve_returns_empty_with_no_entries():
    out = retrieve_relevant_feedback(_job(), {"entries": []}, api_key="key")
    assert out == ""


def test_retrieve_returns_empty_with_no_api_key():
    out = retrieve_relevant_feedback(
        _job(),
        {"entries": [{"text": "[applied] X @ Y", "embedding": [1.0, 0.0]}]},
        api_key="",
    )
    assert out == ""


def test_retrieve_returns_empty_when_job_embed_fails():
    orig = ce.embed_feedback_text
    try:
        ce.embed_feedback_text = lambda t, k: None
        out = retrieve_relevant_feedback(
            _job(),
            {"entries": [{"text": "[applied] X @ Y", "embedding": [1.0, 0.0]}]},
            api_key="key",
        )
        assert out == ""
    finally:
        ce.embed_feedback_text = orig


def test_retrieve_orders_entries_by_similarity_and_truncates_to_top_k():
    orig = ce.embed_feedback_text
    try:
        # Job vector aligns with axis 0. Pick the entry whose embedding is closest.
        ce.embed_feedback_text = lambda t, k: [1.0, 0.0]
        embeddings = {"entries": [
            {"text": "[applied] AXIS1 @ A", "embedding": [0.0, 1.0]},   # orthogonal -> 0.0
            {"text": "[applied] AXIS0 @ B", "embedding": [1.0, 0.0]},   # parallel -> 1.0
            {"text": "[applied] DIAG  @ C", "embedding": [0.7, 0.7]},   # ~0.7
            {"text": "[applied] NEG   @ D", "embedding": [-1.0, 0.0]},  # -1.0
        ]}
        out = retrieve_relevant_feedback(_job(), embeddings, api_key="key", top_k=2)
        lines = out.splitlines()
        # First line is the header.
        assert lines[0].startswith("Most similar past feedback entries")
        # Next two are the top-2 ordered by similarity.
        assert "AXIS0" in lines[1]
        assert "DIAG"  in lines[2]
        assert "AXIS1" not in out and "NEG" not in out
    finally:
        ce.embed_feedback_text = orig


def test_retrieve_skips_entries_with_no_embedding():
    orig = ce.embed_feedback_text
    try:
        ce.embed_feedback_text = lambda t, k: [1.0, 0.0]
        embeddings = {"entries": [
            {"text": "[applied] HAS @ A", "embedding": [1.0, 0.0]},
            {"text": "[applied] NONE @ B", "embedding": None},
            {"text": "", "embedding": [0.9, 0.1]},  # empty text — skipped too
        ]}
        out = retrieve_relevant_feedback(_job(), embeddings, api_key="key", top_k=5)
        assert "HAS" in out
        assert "NONE" not in out
    finally:
        ce.embed_feedback_text = orig


def test_retrieve_handles_malformed_embeddings_dict():
    """A None or non-dict input must not raise — degrade to empty."""
    assert retrieve_relevant_feedback(_job(), None, api_key="key") == ""
    assert retrieve_relevant_feedback(_job(), "garbage", api_key="key") == ""


def test_retrieve_uses_default_top_k_constant():
    """RAG_TOP_K is exported so scraper.py can pin the value at the call site."""
    assert isinstance(RAG_TOP_K, int) and RAG_TOP_K > 0


# ---------------------------------------------------------------------------
# RAG threshold constant pins
# ---------------------------------------------------------------------------

def test_rag_threshold_is_a_reasonable_int():
    """The exact number is a product choice (50), but the type/sign matters
    for the routing comparison in scraper.py."""
    assert isinstance(RAG_FEEDBACK_THRESHOLD, int)
    assert RAG_FEEDBACK_THRESHOLD >= 1
