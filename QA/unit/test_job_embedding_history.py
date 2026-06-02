"""Per-user job embedding history tests (Divergence A — multi-user semantic dedup).

Locks load_job_embedding_history / save_job_embedding_history: the Supabase-backed
analog of core_embedding's local embedding_history.json. Uses a tiny fake client
(offline) like the other core_supabase tests.
"""

from pipeline.core_supabase import (
    load_job_embedding_history,
    save_job_embedding_history,
)


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, table, store):
        self.table = table
        self.store = store
        self._eq = {}
        self._gte = {}
        self._upsert = None
        self._on_conflict = None

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._eq[col] = val
        return self

    def gte(self, col, val):
        self._gte[col] = val
        return self

    def upsert(self, rows, on_conflict=None):
        self._upsert = rows
        self._on_conflict = on_conflict
        return self

    def execute(self):
        if self._upsert is not None:
            self.store.setdefault(self.table, []).append(
                {"op": "upsert", "rows": self._upsert, "on_conflict": self._on_conflict}
            )
            if getattr(self.store, "_raise", False):
                raise RuntimeError("simulated outage")
            return _Resp(self._upsert)
        rows = [r for r in self.store.get(self.table, []) if isinstance(r, dict) and "op" not in r]
        for col, val in self._eq.items():
            rows = [r for r in rows if r.get(col) == val]
        # gte on embedded_at (ISO strings compare lexicographically)
        for col, val in self._gte.items():
            rows = [r for r in rows if (r.get(col) or "") >= val]
        return _Resp(rows)


class _Store(dict):
    pass


class _Client:
    def __init__(self, store=None):
        self.store = store if store is not None else _Store()

    def table(self, name):
        return _Query(name, self.store)


# ---------------------------------------------------------------------------
# load_job_embedding_history
# ---------------------------------------------------------------------------

def test_load_returns_flat_url_to_vector_map():
    store = _Store()
    store["job_embeddings"] = [
        {"user_id": "u1", "job_url": "https://a/1", "embedding": [0.1, 0.2], "embedded_at": "2099-01-01T00:00:00+00:00"},
        {"user_id": "u1", "job_url": "https://a/2", "embedding": [0.3, 0.4], "embedded_at": "2099-01-02T00:00:00+00:00"},
        {"user_id": "u2", "job_url": "https://b/1", "embedding": [0.9], "embedded_at": "2099-01-01T00:00:00+00:00"},
    ]
    out = load_job_embedding_history("u1", client=_Client(store))
    assert out == {"https://a/1": [0.1, 0.2], "https://a/2": [0.3, 0.4]}
    assert "https://b/1" not in out  # other user's rows excluded


def test_load_excludes_stale_rows_outside_window():
    store = _Store()
    store["job_embeddings"] = [
        {"user_id": "u1", "job_url": "https://a/recent", "embedding": [0.1], "embedded_at": "2099-01-01T00:00:00+00:00"},
        {"user_id": "u1", "job_url": "https://a/ancient", "embedding": [0.2], "embedded_at": "1999-01-01T00:00:00+00:00"},
    ]
    out = load_job_embedding_history("u1", client=_Client(store))
    assert "https://a/recent" in out
    assert "https://a/ancient" not in out


def test_load_skips_malformed_rows():
    store = _Store()
    store["job_embeddings"] = [
        {"user_id": "u1", "job_url": "https://a/good", "embedding": [0.1], "embedded_at": "2099-01-01T00:00:00+00:00"},
        {"user_id": "u1", "job_url": "https://a/noemb", "embedding": None, "embedded_at": "2099-01-01T00:00:00+00:00"},
        {"user_id": "u1", "job_url": None, "embedding": [0.5], "embedded_at": "2099-01-01T00:00:00+00:00"},
    ]
    out = load_job_embedding_history("u1", client=_Client(store))
    assert out == {"https://a/good": [0.1]}


def test_load_empty_user_returns_empty():
    assert load_job_embedding_history("", client=_Client()) == {}


# ---------------------------------------------------------------------------
# save_job_embedding_history
# ---------------------------------------------------------------------------

def test_save_upserts_with_conflict_key():
    client = _Client()
    n = save_job_embedding_history(
        "u1", {"https://a/1": [0.1, 0.2], "https://a/2": [0.3]}, client=client
    )
    assert n == 2
    ops = client.store.get("job_embeddings", [])
    assert ops and ops[0]["op"] == "upsert"
    assert ops[0]["on_conflict"] == "user_id,job_url"
    urls = {r["job_url"] for r in ops[0]["rows"]}
    assert urls == {"https://a/1", "https://a/2"}
    assert all(r["user_id"] == "u1" for r in ops[0]["rows"])


def test_save_skips_none_vectors():
    client = _Client()
    n = save_job_embedding_history(
        "u1", {"https://a/ok": [0.1], "https://a/bad": None}, client=client
    )
    assert n == 1
    rows = client.store["job_embeddings"][0]["rows"]
    assert [r["job_url"] for r in rows] == ["https://a/ok"]


def test_save_empty_is_noop():
    client = _Client()
    assert save_job_embedding_history("u1", {}, client=client) == 0
    assert save_job_embedding_history("", {"x": [1]}, client=client) == 0
    assert client.store.get("job_embeddings") is None


def test_save_failure_returns_zero_without_raising():
    store = _Store()
    store._raise = True
    client = _Client(store)
    # Must not raise — a failed history write shouldn't kill the run.
    assert save_job_embedding_history("u1", {"https://a/1": [0.1]}, client=client) == 0
