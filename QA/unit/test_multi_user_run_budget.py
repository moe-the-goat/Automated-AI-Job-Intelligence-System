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


# ---------------------------------------------------------------------------
# _discover_keys — auto-discover API accounts from env (base → legacy → _2/_3…)
# ---------------------------------------------------------------------------

import os


class _EnvPatch:
    """Set/clear env vars for the duration of a block, then restore exactly."""

    def __init__(self, present: dict, absent=()):
        self._present = present
        self._absent = list(absent)
        self._saved = {}

    def __enter__(self):
        for k in list(self._present) + self._absent:
            self._saved[k] = os.environ.get(k)
        for k, v in self._present.items():
            os.environ[k] = v
        for k in self._absent:
            os.environ.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def test_discover_keys_base_legacy_and_numbered():
    # base + MULTI_ legacy + _2 + _3, with _4 absent → all four, in order.
    with _EnvPatch(
        {
            "CEREBRAS_API_KEY": "k1",
            "MULTI_CEREBRAS_API_KEY": "k2",
            "CEREBRAS_API_KEY_2": "k3",
            "CEREBRAS_API_KEY_3": "k4",
        },
        absent=["CEREBRAS_API_KEY_4", "CEREBRAS_API_KEY_5"],
    ):
        assert mur._discover_keys("CEREBRAS_API_KEY", "MULTI_CEREBRAS_API_KEY") == "k1,k2,k3,k4"


def test_discover_keys_stops_at_first_gap():
    # _3 is set but _2 is missing → discovery stops after the base; _3 is ignored.
    with _EnvPatch(
        {"GEMINI_EMBED_API_KEY": "e1", "GEMINI_EMBED_API_KEY_3": "e3"},
        absent=["GEMINI_EMBED_API_KEY_2"],
    ):
        assert mur._discover_keys("GEMINI_EMBED_API_KEY") == "e1"


def test_discover_keys_dedupes_identical_values():
    # The same key wired under two names is counted once.
    with _EnvPatch(
        {"GROQ_API_KEY": "same", "MULTI_GROQ_API_KEY": "same"},
        absent=["GROQ_API_KEY_2"],
    ):
        assert mur._discover_keys("GROQ_API_KEY", "MULTI_GROQ_API_KEY") == "same"


def test_discover_keys_single_and_empty():
    with _EnvPatch({"GEMINI_API_KEY": "g1"}, absent=["GEMINI_API_KEY_2"]):
        assert mur._discover_keys("GEMINI_API_KEY") == "g1"
    with _EnvPatch({}, absent=["NOPE_API_KEY", "NOPE_API_KEY_2"]):
        assert mur._discover_keys("NOPE_API_KEY") == ""


# ---------------------------------------------------------------------------
# Sharding — disjoint users + disjoint account slices across parallel shards
# ---------------------------------------------------------------------------

def test_user_shard_is_stable_and_in_range():
    # Same id → same shard every time; always within [0, count).
    for uid in ("u-abc", "11111111-1111-1111-1111-111111111111", "x"):
        s1 = mur._user_shard(uid, 2)
        s2 = mur._user_shard(uid, 2)
        assert s1 == s2
        assert 0 <= s1 < 2


def test_user_shard_partitions_disjointly_and_covers_everyone():
    # Every user lands in exactly one shard; the shards together cover all users.
    users = [f"user-{i}" for i in range(200)]
    count = 3
    buckets = {0: [], 1: [], 2: []}
    for u in users:
        buckets[mur._user_shard(u, count)].append(u)
    # disjoint + complete
    assert sum(len(v) for v in buckets.values()) == len(users)
    seen = set()
    for v in buckets.values():
        assert not (set(v) & seen)  # no overlap
        seen |= set(v)
    # roughly balanced (each bucket gets a non-trivial share of 200)
    assert all(len(v) > 30 for v in buckets.values())


def test_user_shard_count_one_is_zero():
    assert mur._user_shard("anything", 1) == 0


def test_shard_slice_disjoint_no_account_shared():
    keys = "a,b,c,d"
    s0 = mur._shard_slice(keys, 0, 2)
    s1 = mur._shard_slice(keys, 1, 2)
    assert s0 == "a,c"
    assert s1 == "b,d"
    # No account appears in two shards (the whole point — no RPM collision).
    assert not (set(s0.split(",")) & set(s1.split(",")))


def test_shard_slice_no_sharding_returns_all():
    assert mur._shard_slice("a,b,c", 0, 1) == "a,b,c"


def test_shard_slice_single_key_shared_when_unavoidable():
    # One key, two shards: can't split → both get it (degrades, doesn't crash).
    assert mur._shard_slice("only", 0, 2) == "only"
    assert mur._shard_slice("only", 1, 2) == "only"


def test_shard_slice_more_shards_than_keys_no_empty_shard():
    # 2 keys, 3 shards: shard 2 would slice empty → falls back to a key so it can
    # still run (misconfig guard; matrix should be ≤ account count).
    assert mur._shard_slice("a,b", 2, 3) != ""


def test_local_jobs_file_roundtrip():
    import tempfile
    jobs = [{"title": "Eng", "company": "Acme"}, {"title": "Dev", "company": "Beta"}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sub", "local.json")  # nested → exercises makedirs
        mur._dump_local_jobs(p, jobs)
        assert mur._load_local_jobs(p) == jobs


def test_load_local_jobs_missing_file_is_empty():
    assert mur._load_local_jobs("/no/such/file_xyz.json") == []


# ---------------------------------------------------------------------------
# _finalize_run — email-delivery outcome (Tier D) + pre-migration degradation
# ---------------------------------------------------------------------------

class _UpdateQuery:
    def __init__(self, rec, *, reject_email=False):
        self._rec = rec
        self._reject_email = reject_email

    def update(self, payload):
        self._payload = dict(payload)
        return self

    def eq(self, *_a):
        return self

    def execute(self):
        self._rec["attempts"].append(self._payload)
        if self._reject_email and "email_status" in self._payload:
            raise RuntimeError('column "email_status" does not exist')
        self._rec["final"] = self._payload
        return _CountResp([{"id": 1}], with_count=False)


class _UpdateClient:
    def __init__(self, *, reject_email=False):
        self.rec = {"attempts": [], "final": None}
        self._reject_email = reject_email

    def table(self, _name):
        return _UpdateQuery(self.rec, reject_email=self._reject_email)


def test_finalize_run_records_email_status():
    client = _UpdateClient()
    mur._finalize_run(client, 1, status="success", approved=3,
                      email_status="sent", email_error=None)
    assert client.rec["final"]["status"] == "success"
    assert client.rec["final"]["email_status"] == "sent"
    assert client.rec["final"]["email_error"] is None


def test_finalize_run_records_email_failure_with_error():
    client = _UpdateClient()
    mur._finalize_run(client, 1, status="success",
                      email_status="failed", email_error="SMTP 535 auth")
    assert client.rec["final"]["email_status"] == "failed"
    assert client.rec["final"]["email_error"] == "SMTP 535 auth"


def test_finalize_run_degrades_when_email_columns_missing():
    # Pre-migration-0020 schema: the email_status update is rejected, but the run
    # must still finalize via the retry WITHOUT those columns.
    client = _UpdateClient(reject_email=True)
    mur._finalize_run(client, 1, status="success", approved=2,
                      email_status="sent")
    assert len(client.rec["attempts"]) == 2          # first with, retry without
    assert "email_status" not in client.rec["final"] # the retry dropped it
    assert client.rec["final"]["status"] == "success"  # run still finalized


def test_finalize_run_no_email_status_omits_columns():
    # When no email outcome is passed (early-return paths), the columns aren't set.
    client = _UpdateClient()
    mur._finalize_run(client, 1, status="success", approved=0)
    assert "email_status" not in client.rec["final"]


# ---------------------------------------------------------------------------
# _filter_by_min_match — per-user min-match digest threshold (email-only)
# ---------------------------------------------------------------------------

def test_min_match_zero_returns_all():
    import pandas as pd
    df = pd.DataFrame([{"match_percentage": 30}, {"match_percentage": 90}])
    out = mur._filter_by_min_match(df, 0)
    assert len(out) == 2


def test_min_match_filters_below_threshold():
    import pandas as pd
    df = pd.DataFrame([
        {"title": "a", "match_percentage": 40},
        {"title": "b", "match_percentage": 70},
        {"title": "c", "match_percentage": 85},
    ])
    out = mur._filter_by_min_match(df, 70)
    assert sorted(out["title"]) == ["b", "c"]  # 40 dropped; 70 kept (inclusive)


def test_min_match_treats_missing_as_zero():
    import pandas as pd
    df = pd.DataFrame([{"title": "a", "match_percentage": None}, {"title": "b", "match_percentage": 80}])
    out = mur._filter_by_min_match(df, 50)
    assert list(out["title"]) == ["b"]  # NA → 0 → excluded


def test_min_match_empty_frame_is_safe():
    import pandas as pd
    out = mur._filter_by_min_match(pd.DataFrame(), 50)
    assert out.empty
