"""SupabaseJobTracker tests (B7b).

Locks the contract that SupabaseJobTracker is a true drop-in for the
file-backed JobTracker:
  * load() pulls only this user's rows from the last 60 days,
  * is_seen / mark_seen behave like a set,
  * save() bulk-upserts buffered URLs and clears the buffer,
  * upsert failure does NOT lose the buffer (next tick retries),
  * cross-user reads are scoped by user_id.

Tests use a tiny fake-client harness rather than real supabase-py so the
QA suite stays import-light and offline.
"""

from pipeline.core_supabase import SupabaseJobTracker


# ---------------------------------------------------------------------------
# Fake Supabase client
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Mimics the supabase-py fluent query builder. Records every call so
    tests can assert filter shape (user_id scoping, since-clause, etc.)."""

    def __init__(self, table, ops, store):
        self.table = table
        self.ops = ops
        self.store = store
        self.filters = {}
        self._upsert_payload = None
        self._insert_payload = None
        self._on_conflict = None
        self._ignore_duplicates = False

    def select(self, *_cols):
        self.ops.append(("select", self.table, _cols))
        return self

    def eq(self, col, val):
        self.filters[("eq", col)] = val
        return self

    def gte(self, col, val):
        self.filters[("gte", col)] = val
        return self

    def lte(self, col, val):
        self.filters[("lte", col)] = val
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
        self._upsert_payload = rows
        self._on_conflict = on_conflict
        self._ignore_duplicates = ignore_duplicates
        return self

    def insert(self, rows):
        self._insert_payload = rows
        return self

    def update(self, payload):
        self._update_payload = payload
        return self

    def execute(self):
        if self._upsert_payload is not None:
            self.store.setdefault(self.table, []).append(("upsert", self._upsert_payload, self.filters))
            if getattr(self.store, "_raise_on_upsert", False):
                raise RuntimeError("simulated DB outage")
            return _FakeResp(self._upsert_payload)
        if self._insert_payload is not None:
            self.store.setdefault(self.table, []).append(("insert", self._insert_payload, self.filters))
            return _FakeResp(self._insert_payload)
        # SELECT
        user_id = self.filters.get(("eq", "user_id"))
        since = self.filters.get(("gte", "evaluated_at"))
        rows = list(self.store.get(self.table, []))
        if user_id is not None:
            rows = [r for r in rows if isinstance(r, dict) and r.get("user_id") == user_id]
        if since is not None:
            rows = [r for r in rows if (r.get("evaluated_at") or "9999") >= since]
        return _FakeResp(rows)


class _FakeStore(dict):
    """dict subclass so we can attach behavior flags (_raise_on_upsert)."""


class _FakeClient:
    def __init__(self):
        self.ops = []
        self.store = _FakeStore()

    def table(self, name):
        return _FakeQuery(name, self.ops, self.store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_filters_by_user_and_recency():
    client = _FakeClient()
    client.store["seen_jobs"] = [
        {"user_id": "user-a", "job_url": "https://a/1", "evaluated_at": "2099-01-01T00:00:00+00:00"},
        {"user_id": "user-a", "job_url": "https://a/2", "evaluated_at": "2099-01-02T00:00:00+00:00"},
        {"user_id": "user-b", "job_url": "https://b/1", "evaluated_at": "2099-01-03T00:00:00+00:00"},
        {"user_id": "user-a", "job_url": "https://a/old", "evaluated_at": "1999-01-01T00:00:00+00:00"},
    ]

    tracker = SupabaseJobTracker("user-a", client=client)
    assert tracker.is_seen("https://a/1")
    assert tracker.is_seen("https://a/2")
    assert not tracker.is_seen("https://b/1"), "must not see other users' rows"
    assert not tracker.is_seen("https://a/old"), "stale row outside the 60-day window must be excluded"


def test_mark_seen_then_save_upserts_only_new_rows():
    client = _FakeClient()
    tracker = SupabaseJobTracker("user-a", client=client)

    tracker.mark_seen("https://x/1")
    tracker.mark_seen("https://x/1")  # dedup
    tracker.mark_seen("")              # ignored
    tracker.mark_seen(None)            # ignored
    tracker.mark_seen("https://x/2")

    assert tracker.is_seen("https://x/1")
    assert tracker.is_seen("https://x/2")
    assert not tracker.is_seen("https://x/3")

    tracker.save()
    upserts = [op for op in client.store.get("seen_jobs", []) if op[0] == "upsert"]
    assert len(upserts) == 1
    payload = upserts[0][1]
    assert {row["job_url"] for row in payload} == {"https://x/1", "https://x/2"}
    assert all(row["user_id"] == "user-a" for row in payload)
    # Buffer must clear after a successful save so a second save is a no-op.
    tracker.save()
    upserts_after = [op for op in client.store.get("seen_jobs", []) if op[0] == "upsert"]
    assert len(upserts_after) == 1, "second save with empty buffer must not re-upsert"


def test_save_failure_preserves_buffer_for_next_tick():
    client = _FakeClient()
    tracker = SupabaseJobTracker("user-a", client=client)
    tracker.mark_seen("https://x/1")

    client.store._raise_on_upsert = True
    tracker.save()  # must not raise
    assert tracker._pending == ["https://x/1"], "failed upsert must not clear the buffer"

    client.store._raise_on_upsert = False
    tracker.save()
    upserts = [op for op in client.store.get("seen_jobs", []) if op[0] == "upsert"]
    assert len(upserts) == 2  # the failing one was still recorded by the fake; success on retry
    payload = upserts[-1][1]
    assert payload[0]["job_url"] == "https://x/1"


def test_construction_requires_user_id():
    raised = False
    try:
        SupabaseJobTracker("", client=_FakeClient())
    except ValueError:
        raised = True
    assert raised, "must reject empty user_id to prevent cross-tenant leaks"
