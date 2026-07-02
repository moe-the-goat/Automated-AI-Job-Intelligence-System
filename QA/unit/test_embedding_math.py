"""core_embedding pure-math helpers: cosine, ranking, hashing, cache I/O.

The embedding API call itself is mocked away in integration tests; here we
just verify the math is right and the cache file format round-trips.
"""
import os
import sys
import types
from pipeline.core_embedding import (
    _cv_hash, _read_cached_embedding, _write_cached_embedding,
    cosine_similarity, rank_by_similarity,
)


def test_cv_hash_is_stable():
    assert _cv_hash("hello world") == _cv_hash("hello world")


def test_cv_hash_differs_for_different_text():
    assert _cv_hash("a") != _cv_hash("b")


def test_cv_hash_length():
    assert len(_cv_hash("anything")) == 16


def test_cv_hash_handles_empty_and_none():
    assert isinstance(_cv_hash(""), str)
    assert isinstance(_cv_hash(None), str)


def test_cosine_self_is_one():
    assert abs(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_handles_none():
    assert cosine_similarity(None, [1, 2]) == 0.0
    assert cosine_similarity([1, 2], None) == 0.0


def test_cosine_handles_zero_vector():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_handles_empty_vector():
    assert cosine_similarity([], [1, 2, 3]) == 0.0


def test_rank_orders_by_similarity():
    cv = [1.0, 0.0, 0.0]
    jobs = [
        [1.0, 0.0, 0.0],   # identical to CV
        [0.7, 0.7, 0.0],   # half-aligned
        [0.0, 1.0, 0.0],   # orthogonal
    ]
    sims = rank_by_similarity(cv, jobs)
    assert sims[0] > sims[1] > sims[2]
    assert abs(sims[2]) < 1e-9


def test_cv_embedding_cache_roundtrip():
    """Writing and reading back the cache should preserve the vector for the same text."""
    import pipeline.core_embedding as _ce
    test_cache_path = "_test_cv_embedding.json"
    original_path = _ce.CV_EMBEDDING_CACHE
    _ce.CV_EMBEDDING_CACHE = test_cache_path
    try:
        text = "the candidate has strong python skills"
        vec = [0.1, 0.2, 0.3]
        _write_cached_embedding(text, vec)
        assert _read_cached_embedding(text) == vec
        # And the cache should miss for a different CV text (hash mismatch).
        assert _read_cached_embedding(text + " mutated") is None
    finally:
        _ce.CV_EMBEDDING_CACHE = original_path
        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)


def test_cv_embedding_cache_is_per_user_no_eviction():
    """Two different CVs must COEXIST in the cache — writing one must not evict
    the other. This is the multi-user fix: the old single-slot file made every
    user re-embed because each write overwrote the previous user's entry."""
    import pipeline.core_embedding as _ce
    test_cache_path = "_test_cv_embedding_multi.json"
    original_path = _ce.CV_EMBEDDING_CACHE
    _ce.CV_EMBEDDING_CACHE = test_cache_path
    try:
        _write_cached_embedding("user A cv text", [0.1, 0.1])
        _write_cached_embedding("user B cv text", [0.9, 0.9])
        # Both still resolve — B's write did not evict A.
        assert _read_cached_embedding("user A cv text") == [0.1, 0.1]
        assert _read_cached_embedding("user B cv text") == [0.9, 0.9]
    finally:
        _ce.CV_EMBEDDING_CACHE = original_path
        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)


def test_cv_embedding_cache_migrates_legacy_single_entry():
    """An existing legacy file ({"cv_hash","embedding"}) must still be readable
    after the switch to the keyed-map format, so no cache is lost on upgrade."""
    import json as _json
    import pipeline.core_embedding as _ce
    test_cache_path = "_test_cv_embedding_legacy.json"
    original_path = _ce.CV_EMBEDDING_CACHE
    _ce.CV_EMBEDDING_CACHE = test_cache_path
    try:
        text = "legacy cv"
        with open(test_cache_path, "w", encoding="utf-8") as f:
            _json.dump({"cv_hash": _cv_hash(text), "embedding": [0.5, 0.5]}, f)
        assert _read_cached_embedding(text) == [0.5, 0.5]
    finally:
        _ce.CV_EMBEDDING_CACHE = original_path
        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)


# ---------------------------------------------------------------------------
# Cross-tick job-embedding cache (Tier 3): persist embeddings by text hash so
# slowly-churning feed jobs aren't re-embedded every tick.
# ---------------------------------------------------------------------------

def test_job_embedding_cache_roundtrip():
    import pipeline.core_embedding as _ce
    path = "_test_job_embed_cache.json"
    orig = _ce.JOB_EMBEDDING_CACHE_FILE
    _ce.JOB_EMBEDDING_CACHE_FILE = path
    try:
        _ce.save_job_embedding_cache({"h1": [0.1, 0.2], "h2": [0.3, 0.4]})
        got = _ce.load_job_embedding_cache()
        assert got == {"h1": [0.1, 0.2], "h2": [0.3, 0.4]}
        assert "h3" not in got
    finally:
        _ce.JOB_EMBEDDING_CACHE_FILE = orig
        if os.path.exists(path):
            os.remove(path)


def test_job_embedding_cache_missing_file_is_empty():
    import pipeline.core_embedding as _ce
    orig = _ce.JOB_EMBEDDING_CACHE_FILE
    _ce.JOB_EMBEDDING_CACHE_FILE = "_test_job_embed_absent.json"
    try:
        assert _ce.load_job_embedding_cache() == {}
    finally:
        _ce.JOB_EMBEDDING_CACHE_FILE = orig


def test_job_embedding_cache_ttl_prunes_by_first_seen():
    """An entry rolls off ~TTL days after it FIRST appeared, even though the file
    is reloaded every tick — the original timestamp is preserved, not refreshed."""
    from datetime import datetime, timezone, timedelta
    import pipeline.core_embedding as _ce
    path = "_test_job_embed_ttl.json"
    orig = _ce.JOB_EMBEDDING_CACHE_FILE
    _ce.JOB_EMBEDDING_CACHE_FILE = path
    try:
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        _ce.save_job_embedding_cache({"old": [0.1]}, now=t0)
        # Next tick, 20 days later: seed (loads 'old') + a fresh 'new', then re-save.
        seeded = _ce.load_job_embedding_cache()
        seeded["new"] = [0.9]
        _ce.save_job_embedding_cache(seeded, now=t0 + timedelta(days=20))
        got = _ce.load_job_embedding_cache()
        assert "new" in got          # within the 14-day window
        assert "old" not in got      # first seen 20 days ago -> pruned
    finally:
        _ce.JOB_EMBEDDING_CACHE_FILE = orig
        if os.path.exists(path):
            os.remove(path)


def test_job_embedding_cache_caps_entries():
    import pipeline.core_embedding as _ce
    path = "_test_job_embed_cap.json"
    orig_path = _ce.JOB_EMBEDDING_CACHE_FILE
    orig_max = _ce.JOB_EMBEDDING_CACHE_MAX_ENTRIES
    _ce.JOB_EMBEDDING_CACHE_FILE = path
    _ce.JOB_EMBEDDING_CACHE_MAX_ENTRIES = 3
    try:
        _ce.save_job_embedding_cache({f"h{i}": [float(i)] for i in range(10)})
        assert len(_ce.load_job_embedding_cache()) == 3
    finally:
        _ce.JOB_EMBEDDING_CACHE_FILE = orig_path
        _ce.JOB_EMBEDDING_CACHE_MAX_ENTRIES = orig_max
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# attach_similarity with region/trust weighting (2026-05-17)
# ---------------------------------------------------------------------------
# Ranking is now driven by weighted_score = similarity * region * trust, not by
# raw similarity. These tests use no real Gemini calls — they bypass embedding
# entirely (cv_text=None forces the no-API-key path that returns 0.0 for all
# similarities) and then verify the weighted-score column is computed correctly
# alongside the raw similarity column.

import pandas as pd
from pipeline.core_embedding import attach_similarity


def test_attach_similarity_no_api_key_yields_zero_similarity_but_weighted_column():
    """Without an API key we still need both columns present for downstream code."""
    df = pd.DataFrame([
        {"title": "Engineer", "location": "Berlin, Germany", "description": "Build great stuff.", "job_url": "u1"},
        {"title": "Engineer", "location": "Bangalore, India", "description": "Build great stuff.", "job_url": "u2"},
    ])
    out, embeddings = attach_similarity(df, cv_text="some cv text", api_key="")
    assert "similarity" in out.columns
    assert "weighted_score" in out.columns
    # No similarity computed -> all zeros, and no embeddings collected.
    assert (out["similarity"] == 0.0).all()
    assert (out["weighted_score"] == 0.0).all()
    assert embeddings == {}


def test_attach_similarity_handles_empty_dataframe():
    """An empty input dataframe must come back empty (not crash)."""
    out, embeddings = attach_similarity(pd.DataFrame(), cv_text="x", api_key="")
    assert out.empty
    assert embeddings == {}


def _patched_attach_similarity(df, fixed_vec=None):
    """Helper: run attach_similarity with the embedding calls stubbed to return
    a constant vector for every row, so cosine similarity is identical and
    sort order is driven purely by region/trust/role weights.

    Restores the originals in a finally so other tests aren't affected.
    Returns just the dataframe (most tests don't need the embeddings dict).
    """
    import pipeline.core_embedding as ce
    if fixed_vec is None:
        fixed_vec = [1.0, 0.0]
    orig_get_cv = ce.get_cv_embedding
    orig_embed_jobs = ce.embed_jobs
    try:
        ce.get_cv_embedding = lambda text, key: fixed_vec
        ce.embed_jobs = lambda rows, key, throttle_seconds=0, **_: [fixed_vec] * len(rows)
        df_out, _ = attach_similarity(df, cv_text="cv", api_key="fake-key")
        return df_out
    finally:
        ce.get_cv_embedding = orig_get_cv
        ce.embed_jobs = orig_embed_jobs


def test_weighted_score_orders_eu_above_india_at_equal_similarity():
    """The key behaviour change: EU jobs sort above India jobs when raw
    similarities are identical."""
    df = pd.DataFrame([
        {"title": "Engineer", "location": "Bangalore, India", "description": "x"},
        {"title": "Engineer", "location": "Berlin, Germany", "description": "x"},
        {"title": "Engineer", "location": "Worldwide", "description": "x"},
    ])
    out = _patched_attach_similarity(df)
    # All raw similarities should be equal (~1.0)
    assert all(abs(s - 1.0) < 1e-6 for s in out["similarity"])
    # But weighted scores reflect region tiers:
    #   Worldwide (highly_preferred, 1.30) > Germany (preferred, 1.15) > India (deweighted, 0.70)
    locations_in_order = out["location"].tolist()
    assert locations_in_order[0] == "Worldwide"
    assert locations_in_order[1] == "Berlin, Germany"
    assert locations_in_order[2] == "Bangalore, India"


def test_trusted_company_boost_stacks_with_region_weight():
    """A trusted EU company should rank above an untrusted EU company at equal similarity."""
    df = pd.DataFrame([
        {"title": "Engineer", "company": "Random Co", "location": "Berlin", "pre_flagged_trusted": False, "description": "x"},
        {"title": "Engineer", "company": "Anthropic", "location": "Berlin", "pre_flagged_trusted": True,  "description": "x"},
    ])
    out = _patched_attach_similarity(df)
    # The trusted row should now be on top.
    assert out.iloc[0]["company"] == "Anthropic"


def test_weighted_score_demotes_india_below_worldwide_even_with_higher_raw_similarity():
    """Demonstrate that the weighting bridges meaningful similarity gaps.

    With weights 1.30 (Worldwide) vs 0.70 (India), India needs similarity ~1.86x
    higher to outrank Worldwide. We construct two rows whose raw similarities
    are 1.0 (India) and 0.6 (Worldwide) and confirm Worldwide WINS because
    0.6 * 1.30 = 0.78 vs 1.0 * 0.70 = 0.70.
    """
    import pipeline.core_embedding as ce
    cv_vec = [1.0, 0.0]
    india_vec = [1.0, 0.0]
    world_vec = [0.6, 0.8]                # cos = 0.6
    orig_get = ce.get_cv_embedding
    orig_embed = ce.embed_jobs
    try:
        ce.get_cv_embedding = lambda t, k: cv_vec
        ce.embed_jobs = lambda rows, k, throttle_seconds=0, **_: [
            india_vec if "india" in str(r.get("location", "")).lower() else world_vec
            for r in rows
        ]
        df = pd.DataFrame([
            {"title": "Engineer", "location": "Bangalore, India", "description": "x"},
            {"title": "Engineer", "location": "Worldwide", "description": "x"},
        ])
        out, _ = attach_similarity(df, cv_text="cv", api_key="fake-key")
    finally:
        ce.get_cv_embedding = orig_get
        ce.embed_jobs = orig_embed

    # Worldwide should rank first despite lower raw similarity.
    assert out.iloc[0]["location"] == "Worldwide"
    # And the similarity column should still reflect the raw 0.6 vs 1.0.
    assert out.iloc[0]["similarity"] < out.iloc[1]["similarity"]


def test_attach_similarity_returns_url_to_embedding_dict():
    """attach_similarity must yield {url: embedding} for callers doing semantic dedup."""
    import pipeline.core_embedding as ce
    orig_get = ce.get_cv_embedding
    orig_embed = ce.embed_jobs
    try:
        ce.get_cv_embedding = lambda t, k: [1.0, 0.0]
        ce.embed_jobs = lambda rows, k, throttle_seconds=0, **_: [[0.1, 0.2]] * len(rows)
        df = pd.DataFrame([
            {"title": "Engineer", "job_url": "https://a/1", "description": "x"},
            {"title": "Engineer", "job_url": "https://a/2", "description": "x"},
        ])
        _, embeddings = attach_similarity(df, cv_text="cv", api_key="fake-key")
    finally:
        ce.get_cv_embedding = orig_get
        ce.embed_jobs = orig_embed
    assert set(embeddings.keys()) == {"https://a/1", "https://a/2"}
    assert embeddings["https://a/1"] == [0.1, 0.2]


def test_attach_similarity_skips_urls_with_failed_embeddings():
    """If the embedding API returns None for a row, that URL is omitted from the dict."""
    import pipeline.core_embedding as ce
    orig_get = ce.get_cv_embedding
    orig_embed = ce.embed_jobs
    try:
        ce.get_cv_embedding = lambda t, k: [1.0, 0.0]
        ce.embed_jobs = lambda rows, k, throttle_seconds=0, **_: [[0.1, 0.2], None]
        df = pd.DataFrame([
            {"title": "Engineer", "job_url": "https://a/ok", "description": "x"},
            {"title": "Engineer", "job_url": "https://a/fail", "description": "x"},
        ])
        _, embeddings = attach_similarity(df, cv_text="cv", api_key="fake-key")
    finally:
        ce.get_cv_embedding = orig_get
        ce.embed_jobs = orig_embed
    assert "https://a/ok" in embeddings
    assert "https://a/fail" not in embeddings


# ---------------------------------------------------------------------------
# Shared per-tick job-embedding cache — embed each unique job ONCE per tick,
# even when several users see it (the scaling win: cuts embedding RPD + time).
# ---------------------------------------------------------------------------

class _FakeEmbedResp:
    def __init__(self, vec):
        self.embeddings = [type("E", (), {"values": vec})()]


def _stub_genai(monkeysaved):
    """Stub google.genai so embed_jobs can build a Client without the real SDK."""
    fake_genai = types.SimpleNamespace(Client=lambda api_key=None: object())
    google_mod = types.ModuleType("google")
    google_mod.genai = fake_genai
    for name, mod in (("google", google_mod), ("google.genai", fake_genai)):
        monkeysaved[name] = sys.modules.get(name)
        sys.modules[name] = mod


def _restore_modules(monkeysaved):
    for name, mod in monkeysaved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_embed_jobs_shared_cache_embeds_identical_job_once():
    import pipeline.core_embedding as ce
    saved = {}
    _stub_genai(saved)
    orig = ce._embed_text
    calls = {"n": 0}

    def counting_embed(client, text, model=ce.EMBED_MODEL):
        calls["n"] += 1
        return [0.1, 0.2, 0.3]

    ce._embed_text = counting_embed
    try:
        cache: dict = {}
        # User 1 sees two rows with IDENTICAL text → 1 embed (2nd is a hit).
        rows = [
            {"title": "Backend Engineer", "description": "Build APIs."},
            {"title": "Backend Engineer", "description": "Build APIs."},
        ]
        out1 = ce.embed_jobs(rows, "key", throttle_seconds=0, cache=cache)
        # User 2 (same tick, same shared cache) sees the SAME job → 0 new embeds.
        out2 = ce.embed_jobs(
            [{"title": "Backend Engineer", "description": "Build APIs."}],
            "key", throttle_seconds=0, cache=cache,
        )
    finally:
        ce._embed_text = orig
        _restore_modules(saved)

    assert calls["n"] == 1                       # embedded once across 3 identical rows
    assert out1[0] == out1[1] == out2[0] == [0.1, 0.2, 0.3]


def test_embed_jobs_rotates_across_embedding_accounts():
    # Two embedding accounts → distinct job texts are spread across both clients,
    # doubling daily-RPD headroom (the embedding bottleneck).
    import pipeline.core_embedding as ce
    saved = {}
    created_keys = []
    fake_genai = types.SimpleNamespace(
        Client=lambda api_key=None: created_keys.append(api_key) or object()
    )
    google_mod = types.ModuleType("google")
    google_mod.genai = fake_genai
    for name, mod in (("google", google_mod), ("google.genai", fake_genai)):
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    orig = ce._embed_text
    ce._embed_text = lambda client, text, model=ce.EMBED_MODEL: [0.1]
    try:
        rows = [
            {"title": "A", "description": "1"},
            {"title": "B", "description": "2"},
            {"title": "C", "description": "3"},
            {"title": "D", "description": "4"},
        ]
        ce.embed_jobs(rows, "key1,key2", throttle_seconds=0)
    finally:
        ce._embed_text = orig
        _restore_modules(saved)
    # Both accounts were used (a client was built for each).
    assert set(created_keys) == {"key1", "key2"}


def test_embed_jobs_without_cache_embeds_every_row():
    # cache=None preserves the old behavior: every row is embedded.
    import pipeline.core_embedding as ce
    saved = {}
    _stub_genai(saved)
    orig = ce._embed_text
    calls = {"n": 0}
    ce._embed_text = lambda client, text, model=ce.EMBED_MODEL: (
        calls.__setitem__("n", calls["n"] + 1) or [1.0]
    )
    try:
        rows = [
            {"title": "Same", "description": "Same"},
            {"title": "Same", "description": "Same"},
        ]
        ce.embed_jobs(rows, "key", throttle_seconds=0)  # no cache
    finally:
        ce._embed_text = orig
        _restore_modules(saved)
    assert calls["n"] == 2                       # both embedded (no dedup without a cache)
