"""multi_user_runner RAG vs digest provider selection + empty-corpus guard.

The 2026-06-11 cutover run logged "RAG mode (75 entries)" while the embedding
corpus had silently loaded EMPTY (the feedback↔feedback_embeddings FK was
missing from PostgREST's schema cache). Every verdict got zero feedback
context. _build_preferences_provider now distinguishes:
  * >= threshold AND embeddings present  -> RAG, info log
  * >= threshold BUT corpus empty        -> RAG, ERROR log (load failure, not "no feedback")
  * <  threshold                          -> digest

We avoid pytest's `monkeypatch` fixture because the project's QA runner calls
test functions with no args — manual swap + finally restore (see test_websearch.py).
"""

import logging

import multi_user_runner as mur


def _swap(**attrs):
    """Swap named attributes on the runner module; return a restore() callable."""
    originals = {name: getattr(mur, name) for name in attrs}
    for name, val in attrs.items():
        setattr(mur, name, val)

    def restore():
        for name, val in originals.items():
            setattr(mur, name, val)

    return restore


def test_rag_mode_with_loaded_embeddings():
    restore = _swap(
        ensure_feedback_embeddings=lambda uid, key: mur.RAG_FEEDBACK_THRESHOLD + 25,
        load_feedback_embeddings=lambda uid: {"entries": [
            {"text": "applied: Backend dev", "embedding": [0.1, 0.2]},
        ]},
        retrieve_relevant_feedback=lambda row, emb, key, top_k=0: "CONTEXT",
    )
    try:
        provider, mode = mur._build_preferences_provider("u1", "key")
        assert mode == "rag"
        # The provider must actually route through retrieval.
        assert provider({"title": "x"}) == "CONTEXT"
    finally:
        restore()


def test_rag_mode_empty_corpus_logs_error(caplog=None):
    """Past threshold but the embedding corpus loaded empty => ERROR, still rag mode."""
    restore = _swap(
        ensure_feedback_embeddings=lambda uid, key: mur.RAG_FEEDBACK_THRESHOLD + 25,
        load_feedback_embeddings=lambda uid: {"entries": []},   # load failure / FK missing
        retrieve_relevant_feedback=lambda row, emb, key, top_k=0: "",
    )

    # Capture logs from the runner without relying on the caplog fixture.
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    mur.logger.addHandler(handler)
    prev_level = mur.logger.level
    mur.logger.setLevel(logging.DEBUG)
    try:
        provider, mode = mur._build_preferences_provider("u1", "key")
        assert mode == "rag"  # mode is still rag; it's the corpus that's empty
        # An ERROR must have been emitted naming the empty corpus.
        errors = [r for r in records if r.levelno >= logging.ERROR]
        assert any("EMPTY" in r.getMessage() for r in errors), \
            "empty RAG corpus must log an ERROR, not a cheerful info line"
    finally:
        mur.logger.removeHandler(handler)
        mur.logger.setLevel(prev_level)
        restore()


def test_digest_mode_below_threshold():
    restore = _swap(
        ensure_feedback_embeddings=lambda uid, key: mur.RAG_FEEDBACK_THRESHOLD - 1,
        load_candidate_preferences=lambda uid: "DIGEST PROFILE",
    )
    try:
        provider, mode = mur._build_preferences_provider("u1", "key")
        assert mode == "digest"
        assert provider({"title": "anything"}) == "DIGEST PROFILE"
    finally:
        restore()
