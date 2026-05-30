"""multi_user_runner scheduling/lock helpers (B7 run-lock).

The run lock pushes next_run_at forward at the START of a user's run so an
overlapping cron tick can't re-select an in-flight user. These pure helpers
compute the claim time (now + cadence) and the post-failure retry time.
"""

from datetime import datetime, timedelta, timezone

import multi_user_runner as mur

_NOW = datetime(2026, 5, 30, 9, 0, 0, tzinfo=timezone.utc)


def test_compute_next_run_adds_cadence():
    assert mur._compute_next_run(24, now=_NOW) == _NOW + timedelta(hours=24)
    assert mur._compute_next_run(168, now=_NOW) == _NOW + timedelta(hours=168)


def test_compute_next_run_floors_at_one_hour():
    # Guards against a 0/negative cadence pinning a user as perpetually due.
    assert mur._compute_next_run(0, now=_NOW) == _NOW + timedelta(hours=1)
    assert mur._compute_next_run(-5, now=_NOW) == _NOW + timedelta(hours=1)


def test_compute_next_run_coerces_stringy_frequency():
    assert mur._compute_next_run("24", now=_NOW) == _NOW + timedelta(hours=24)


def test_compute_retry_is_short_backoff():
    assert mur._compute_retry(now=_NOW) == _NOW + timedelta(hours=mur.FAILURE_RETRY_HOURS)


def test_retry_is_sooner_than_a_daily_cadence():
    # The whole point: a failed daily user retries well before 24h.
    assert mur._compute_retry(now=_NOW) < mur._compute_next_run(24, now=_NOW)
