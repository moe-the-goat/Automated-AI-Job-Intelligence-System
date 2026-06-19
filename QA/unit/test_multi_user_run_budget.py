"""multi_user_runner daily run-budget + manual-trigger plumbing.

Locks the data layer behind the dashboard's "Run now" / 2-runs-a-day feature:
  * the local-midnight (Asia/Jerusalem) day boundary the budget resets on;
  * _runs_used_today counts runs since that boundary and fails CLOSED;
  * _insert_run stamps run_trigger and degrades gracefully pre-migration;
  * main()'s --manual flag only takes effect with --user-id.

Pure-helper tests use fixed `now`; the count/insert tests use a tiny fake
Supabase client modeling only the calls these functions make.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import multi_user_runner as mur


_JLM = ZoneInfo("Asia/Jerusalem")


# ---------------------------------------------------------------------------
# Day-boundary helpers
# ---------------------------------------------------------------------------

def test_budget_day_start_is_local_midnight_in_utc():
    # 2026-06-13 20:14 Jerusalem (IDT, UTC+3) → that day's local midnight is
    # 2026-06-13 00:00 +03:00 == 2026-06-12 21:00 UTC.
    now = datetime(2026, 6, 13, 20, 14, tzinfo=_JLM).astimezone(timezone.utc)
    start = mur._budget_day_start_utc(now=now)
    assert start == datetime(2026, 6, 12, 21, 0, tzinfo=timezone.utc)
    # And it's genuinely local midnight when viewed in Jerusalem.
    local = start.astimezone(_JLM)
    assert (local.hour, local.minute, local.second) == (0, 0, 0)


def test_budget_day_start_just_after_local_midnight():
    # 00:30 Jerusalem should map to the SAME day's midnight (30 min earlier),
    # not the previous day's — i.e. a fresh budget right after midnight.
    now = datetime(2026, 6, 13, 0, 30, tzinfo=_JLM).astimezone(timezone.utc)
    start = mur._budget_day_start_utc(now=now)
    assert start == datetime(2026, 6, 13, 0, 0, tzinfo=_JLM).astimezone(timezone.utc)


def test_budget_day_str_is_local_calendar_date_not_utc():
    # 2026-06-18 08:00 Jerusalem. The budget day stamp must be 2026-06-18 (the
    # LOCAL calendar date), NOT 2026-06-17 — which is what
    # _budget_day_start_utc().date() wrongly returned (the UTC date of the
    # Jerusalem-midnight instant, always a day behind). This off-by-one is the
    # bug that left the dashboard's "Today" LLM-usage view empty all day.
    now = datetime(2026, 6, 18, 8, 0, tzinfo=_JLM).astimezone(timezone.utc)
    assert mur._budget_day_str(now=now) == "2026-06-18"
    # Sanity: it diverges from the old UTC-date behaviour at this instant.
    assert mur._budget_day_start_utc(now=now).date().isoformat() == "2026-06-17"


def test_budget_day_str_just_after_local_midnight():
    # 00:30 Jerusalem on the 18th is still the 18th's budget day.
    now = datetime(2026, 6, 18, 0, 30, tzinfo=_JLM).astimezone(timezone.utc)
    assert mur._budget_day_str(now=now) == "2026-06-18"


def test_next_budget_day_start_is_24h_after_today_start():
    now = datetime(2026, 6, 13, 20, 14, tzinfo=_JLM).astimezone(timezone.utc)
    today = mur._budget_day_start_utc(now=now)
    nxt = mur._next_budget_day_start_utc(now=now)
    assert nxt == today + timedelta(days=1)
    # The reset instant is in the future relative to now.
    assert nxt > now


# ---------------------------------------------------------------------------
# _runs_used_today
# ---------------------------------------------------------------------------

class _CountQuery:
    """Models client.table('runs').select('id', count='exact').eq().gte().execute()."""

    def __init__(self, rows, *, raise_on_execute=False):
        self._rows = rows
        self._eq = {}
        self._gte = {}
        self._count_requested = False
        self._raise = raise_on_execute

    def select(self, *_a, **kw):
        self._count_requested = kw.get("count") == "exact"
        return self

    def eq(self, col, val):
        self._eq[col] = val
        return self

    def gte(self, col, val):
        self._gte[col] = val
        return self

    def execute(self):
        if self._raise:
            raise RuntimeError("simulated DB outage")
        rows = [r for r in self._rows if all(r.get(c) == v for c, v in self._eq.items())]
        for col, val in self._gte.items():
            rows = [r for r in rows if r.get(col) >= val]
        return _CountResp(rows, self._count_requested)


class _CountResp:
    def __init__(self, rows, with_count):
        self.data = rows
        self.count = len(rows) if with_count else None


class _CountClient:
    def __init__(self, rows, *, raise_on_execute=False):
        self._rows = rows
        self._raise = raise_on_execute
        self.last_query = None

    def table(self, _name):
        self.last_query = _CountQuery(self._rows, raise_on_execute=self._raise)
        return self.last_query


def _run_row(user_id, started_at):
    return {"id": 1, "user_id": user_id, "started_at": started_at.isoformat()}


def test_runs_used_today_counts_only_since_local_midnight():
    now = datetime(2026, 6, 13, 20, 0, tzinfo=_JLM).astimezone(timezone.utc)
    start = mur._budget_day_start_utc(now=now)
    rows = [
        _run_row("u1", start + timedelta(hours=1)),   # today — counts
        _run_row("u1", start + timedelta(hours=5)),   # today — counts
        _run_row("u1", start - timedelta(minutes=5)), # yesterday — excluded
        _run_row("u2", start + timedelta(hours=2)),   # other user — excluded
    ]
    client = _CountClient(rows)
    assert mur._runs_used_today(client, "u1", now=now) == 2


def test_runs_used_today_zero_when_no_runs():
    now = datetime(2026, 6, 13, 12, 0, tzinfo=_JLM).astimezone(timezone.utc)
    assert mur._runs_used_today(_CountClient([]), "u1", now=now) == 0


def test_runs_used_today_fails_closed_on_db_error():
    # A transient DB failure must NOT hand out free runs — return the cap so
    # the caller skips the run rather than over-spending.
    client = _CountClient([], raise_on_execute=True)
    assert mur._runs_used_today(client, "u1") == mur.MAX_RUNS_PER_DAY


# ---------------------------------------------------------------------------
# _insert_run trigger stamping
# ---------------------------------------------------------------------------

class _InsertQuery:
    def __init__(self, recorder, *, fail_with_trigger=False):
        self._rec = recorder
        self._fail_with_trigger = fail_with_trigger

    def insert(self, payload):
        self._rec["payloads"].append(payload)
        self._payload = payload
        return self

    def execute(self):
        if self._fail_with_trigger and "run_trigger" in self._payload:
            raise RuntimeError('column "run_trigger" does not exist')
        return _CountResp([{"id": 99}], with_count=False)


class _InsertClient:
    def __init__(self, *, fail_with_trigger=False):
        self.recorder = {"payloads": []}
        self._fail_with_trigger = fail_with_trigger

    def table(self, _name):
        return _InsertQuery(self.recorder, fail_with_trigger=self._fail_with_trigger)


def test_insert_run_stamps_trigger():
    client = _InsertClient()
    rid = mur._insert_run(client, "u1", trigger="manual")
    assert rid == 99
    assert client.recorder["payloads"][-1]["run_trigger"] == "manual"


def test_insert_run_defaults_to_scheduled():
    client = _InsertClient()
    mur._insert_run(client, "u1")
    assert client.recorder["payloads"][-1]["run_trigger"] == "scheduled"


def test_insert_run_coerces_unknown_trigger_to_scheduled():
    client = _InsertClient()
    mur._insert_run(client, "u1", trigger="garbage")
    assert client.recorder["payloads"][-1]["run_trigger"] == "scheduled"


def test_insert_run_retries_without_trigger_when_column_missing():
    # Pre-migration 0014: the run_trigger insert fails, but the run must still
    # be created via the column-less retry so the pipeline keeps working.
    client = _InsertClient(fail_with_trigger=True)
    rid = mur._insert_run(client, "u1", trigger="manual")
    assert rid == 99
    payloads = client.recorder["payloads"]
    assert "run_trigger" in payloads[0]            # first attempt had it
    assert "run_trigger" not in payloads[1]        # retry dropped it


# ---------------------------------------------------------------------------
# Budget constant sanity
# ---------------------------------------------------------------------------

def test_max_runs_per_day_is_two():
    assert mur.MAX_RUNS_PER_DAY == 2


# ---------------------------------------------------------------------------
# _budget_allows_run — admin override bypasses the cap
# ---------------------------------------------------------------------------

def test_budget_allows_run_under_cap():
    assert mur._budget_allows_run(0) is True
    assert mur._budget_allows_run(mur.MAX_RUNS_PER_DAY - 1) is True


def test_budget_allows_run_blocks_at_cap_without_override():
    assert mur._budget_allows_run(mur.MAX_RUNS_PER_DAY) is False
    assert mur._budget_allows_run(mur.MAX_RUNS_PER_DAY + 5) is False


def test_admin_override_bypasses_cap():
    # The whole point of the admin "Force run": run even when the user is maxed.
    assert mur._budget_allows_run(mur.MAX_RUNS_PER_DAY, admin_override=True) is True
    assert mur._budget_allows_run(99, admin_override=True) is True
