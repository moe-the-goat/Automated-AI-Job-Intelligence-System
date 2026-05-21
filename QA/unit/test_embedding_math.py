"""core_embedding pure-math helpers: cosine, ranking, hashing, cache I/O.

The embedding API call itself is mocked away in integration tests; here we
just verify the math is right and the cache file format round-trips.
"""
import os
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
        ce.embed_jobs = lambda rows, key, throttle_seconds=0: [fixed_vec] * len(rows)
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
        ce.embed_jobs = lambda rows, k, throttle_seconds=0: [
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
        ce.embed_jobs = lambda rows, k, throttle_seconds=0: [[0.1, 0.2]] * len(rows)
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
        ce.embed_jobs = lambda rows, k, throttle_seconds=0: [[0.1, 0.2], None]
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
