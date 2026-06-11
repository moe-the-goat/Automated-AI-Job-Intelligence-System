"""multi_user_runner local-jobs cache (B11 cutover — local coverage in multi-user).

The Palestinian local market is identical for every user, so the runner scrapes
it ONCE per cron tick (_LocalJobsCache) and hands the same raw list to each
user's pipeline, where per-user CV ranking + RAG personalizes what surfaces.

Locking the contract:
  * collection runs at most once across many .get() calls (memoized)
  * a collection failure degrades to [] — it must NEVER raise into the run loop
    (one dead local source can't take down every user's email)
  * an empty company list still returns a (possibly empty) list, not None

We avoid pytest's `monkeypatch` fixture because the project's QA runner calls
test functions with no args — so we manually swap the attribute and restore it
in a finally block (matching test_websearch.py's convention).
"""

import multi_user_runner as mur


def _swap_collect(fn):
    """Swap collect_local_raw_jobs on the runner; return a restore() callable."""
    orig = mur.collect_local_raw_jobs
    mur.collect_local_raw_jobs = fn

    def restore():
        mur.collect_local_raw_jobs = orig

    return restore


def test_local_cache_collects_only_once():
    calls = {"n": 0}

    def fake_collect(*, gemini_key=None, lookback_days=None):
        calls["n"] += 1
        jobs = [{"title": "Local Dev", "job_url": "https://x/local-1", "source": "ddg_linkedin"}]
        stats = {"ats_jobs": 0, "ddg_jobs": 1, "jobspy_jobs": 0,
                 "telegram_jobs": 0, "jobsps_jobs": 0}
        return jobs, stats

    restore = _swap_collect(fake_collect)
    try:
        cache = mur._LocalJobsCache()
        first = cache.get()
        second = cache.get()
        third = cache.get()

        assert calls["n"] == 1, "local collection must run once per tick, not per user"
        assert first is second is third, "all users share the same cached list object"
        assert first == [{"title": "Local Dev", "job_url": "https://x/local-1", "source": "ddg_linkedin"}]
    finally:
        restore()


def test_local_cache_degrades_to_empty_on_failure():
    def boom(*, gemini_key=None, lookback_days=None):
        raise RuntimeError("simulated local source outage")

    restore = _swap_collect(boom)
    try:
        cache = mur._LocalJobsCache()
        # Must NOT raise — a dead local source can't break the per-user run loop.
        result = cache.get()
        assert result == []
        # And the failure result is memoized too (no retry storm within the tick).
        assert cache.get() == []
    finally:
        restore()


def test_local_cache_returns_empty_list_not_none():
    def empty(*, gemini_key=None, lookback_days=None):
        return [], {"ats_jobs": 0, "ddg_jobs": 0, "jobspy_jobs": 0,
                    "telegram_jobs": 0, "jobsps_jobs": 0}

    restore = _swap_collect(empty)
    try:
        cache = mur._LocalJobsCache()
        result = cache.get()
        assert result == []
        assert result is not None
    finally:
        restore()


def test_include_local_sources_flag_is_on():
    """Cutover requires local coverage in the multi-user pipeline to be enabled."""
    assert mur.INCLUDE_LOCAL_SOURCES is True
