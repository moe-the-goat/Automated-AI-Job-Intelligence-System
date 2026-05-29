"""multi_user_runner._load_due_users tests (B7).

Locks the two-query load that replaced the invalid PostgREST embed of
search_queries under preferences (PGRST200 — no FK between them). Also pins
the gating: whitelist, missing CV, and no active searches each drop a user;
search_queries are merged in by user_id.
"""

import multi_user_runner as mur


# ---------------------------------------------------------------------------
# Fake Supabase client — models the two queries _load_due_users makes.
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, table, store):
        self.table = table
        self.store = store
        self._eq = {}
        self._in = {}

    def select(self, *_a, **_k):
        return self

    def eq(self, col, val):
        self._eq[col] = val
        return self

    def lte(self, _col, _val):
        return self  # next_run_at gating not exercised here

    def in_(self, col, vals):
        self._in[col] = set(vals)
        return self

    def execute(self):
        rows = list(self.store.get(self.table, []))
        for col, val in self._eq.items():
            rows = [r for r in rows if r.get(col) == val]
        for col, vals in self._in.items():
            rows = [r for r in rows if r.get(col) in vals]
        return _Resp(rows)


class _Client:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        return _Query(name, self.store)


def _pref(uid, *, whitelisted=True, cv="my cv", **over):
    return {
        "user_id": uid,
        "frequency_hours": 24,
        "is_active": True,
        "next_run_at": "2026-05-29T00:00:00+00:00",
        "notification_email": f"{uid}@x.com",
        "ai_eval_top_n": 30,
        "api_hours_old": 72,
        "profiles": {"cv_text": cv, "is_whitelisted": whitelisted},
        **over,
    }


def _search(uid, term="engineer"):
    return {
        "user_id": uid, "search_term": term, "location": "Worldwide",
        "sites": ["linkedin"], "job_type": None, "is_remote": True,
        "results_wanted": 30, "hours_old": 24, "country_indeed": "USA",
        "is_active": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loads_due_user_with_merged_searches():
    client = _Client({
        "preferences": [_pref("u1")],
        "search_queries": [_search("u1", "backend"), _search("u1", "ml")],
    })
    users = mur._load_due_users(client, skip_due_check=True)
    assert len(users) == 1
    u = users[0]
    assert u["user_id"] == "u1"
    assert u["cv_text"] == "my cv"
    assert u["notification_email"] == "u1@x.com"
    assert {s["search_term"] for s in u["search_queries"]} == {"backend", "ml"}


def test_skips_non_whitelisted():
    client = _Client({
        "preferences": [_pref("u1", whitelisted=False)],
        "search_queries": [_search("u1")],
    })
    assert mur._load_due_users(client, skip_due_check=True) == []


def test_skips_missing_cv():
    client = _Client({
        "preferences": [_pref("u1", cv="   ")],
        "search_queries": [_search("u1")],
    })
    assert mur._load_due_users(client, skip_due_check=True) == []


def test_skips_user_without_active_searches():
    client = _Client({
        "preferences": [_pref("u1")],
        "search_queries": [],  # none for u1
    })
    assert mur._load_due_users(client, skip_due_check=True) == []


def test_searches_are_scoped_per_user():
    client = _Client({
        "preferences": [_pref("u1"), _pref("u2")],
        "search_queries": [_search("u1", "a"), _search("u2", "b")],
    })
    users = {u["user_id"]: u for u in mur._load_due_users(client, skip_due_check=True)}
    assert users["u1"]["search_queries"][0]["search_term"] == "a"
    assert users["u2"]["search_queries"][0]["search_term"] == "b"
    assert len(users["u1"]["search_queries"]) == 1


def test_only_user_id_filters_to_one():
    client = _Client({
        "preferences": [_pref("u1"), _pref("u2")],
        "search_queries": [_search("u1"), _search("u2")],
    })
    users = mur._load_due_users(client, only_user_id="u2", skip_due_check=True)
    assert len(users) == 1 and users[0]["user_id"] == "u2"


def test_empty_preferences_returns_empty():
    client = _Client({"preferences": [], "search_queries": []})
    assert mur._load_due_users(client, skip_due_check=True) == []
