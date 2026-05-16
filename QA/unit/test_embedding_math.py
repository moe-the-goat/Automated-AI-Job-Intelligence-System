"""core_embedding pure-math helpers: cosine, ranking, hashing, cache I/O.

The embedding API call itself is mocked away in integration tests; here we
just verify the math is right and the cache file format round-trips.
"""
import os
from core_embedding import (
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
    import core_embedding as _ce
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
