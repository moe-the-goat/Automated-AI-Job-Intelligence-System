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


# --- anchored next-run (no schedule drift on a late fire) -------------------

def test_anchored_keeps_scheduled_time_when_fired_late():
    # Scheduled 10:00, but the tick fires at 11:00. Next run must be 10:00 the
    # NEXT day (anchor + 24h), NOT 11:00 (fired + 24h).
    scheduled = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
    fired_late = datetime(2026, 5, 30, 11, 0, tzinfo=timezone.utc)
    nxt = mur._compute_next_run_anchored(scheduled.isoformat(), 24, now=fired_late)
    assert nxt == datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)


def test_anchored_accepts_a_datetime_anchor_too():
    scheduled = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
    fired_late = datetime(2026, 5, 30, 14, 30, tzinfo=timezone.utc)
    nxt = mur._compute_next_run_anchored(scheduled, 24, now=fired_late)
    assert nxt == datetime(2026, 5, 31, 10, 0, tzinfo=timezone.utc)


def test_anchored_catches_up_when_very_late():
    # If a daily run was missed for ~2 days, the next slot must be the next
    # FUTURE 10:00 on the original cadence, not back-to-back runs.
    scheduled = datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)  # ~2 days + 2h later
    nxt = mur._compute_next_run_anchored(scheduled.isoformat(), 24, now=now)
    assert nxt > now
    assert nxt == datetime(2026, 6, 2, 10, 0, tzinfo=timezone.utc)


def test_anchored_falls_back_to_now_plus_cadence_without_anchor():
    # No usable anchor (manual / first run) → behaves like _compute_next_run.
    assert mur._compute_next_run_anchored(None, 24, now=_NOW) == _NOW + timedelta(hours=24)
    assert mur._compute_next_run_anchored("not-a-date", 24, now=_NOW) == _NOW + timedelta(hours=24)


def test_parse_iso_handles_z_and_offset_and_garbage():
    assert mur._parse_iso("2026-05-30T10:00:00Z") == datetime(2026, 5, 30, 10, 0, tzinfo=timezone.utc)
    assert mur._parse_iso(None) is None
    assert mur._parse_iso("nope") is None
